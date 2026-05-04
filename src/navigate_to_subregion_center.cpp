#include "navigate_to_subregion_center.hpp"

using namespace std::chrono_literals;

NavigateToSubregionCenter::NavigateToSubregionCenter(
    const std::string& name,
    const BT::NodeConfiguration& config,
    rclcpp::Node::SharedPtr node)
    : BT::StatefulActionNode(name, config),
      node_(node),
      state_(NavState::IDLE)
{
    // 在构造期初始化 Action Client，不阻塞行为树
    client_ = rclcpp_action::create_client<NavAction>(node_, "/navigate_to_pose");
}

BT::PortsList NavigateToSubregionCenter::providedPorts()
{
    // 严格定义 PoseStamped 类型的输入端口
    return { BT::InputPort<geometry_msgs::msg::PoseStamped>("goal", "目标位姿") };
}

BT::NodeStatus NavigateToSubregionCenter::onStart()
{
    // 1. 非阻塞检查服务器可用性
    if (!client_->action_server_is_ready()) {
        RCLCPP_ERROR(node_->get_logger(), 
                     "[NavigateToSubregionCenter] Nav2 action server is not ready. Returning FAILURE.");
        return BT::NodeStatus::FAILURE;
    }

    // 2. 从黑板读取目标坐标
    if (!getInput<geometry_msgs::msg::PoseStamped>("goal", current_goal_)) {
        RCLCPP_ERROR(node_->get_logger(), 
                     "[NavigateToSubregionCenter] Failed to retrieve 'goal' from Blackboard.");
        return BT::NodeStatus::FAILURE;
    }

    RCLCPP_INFO(node_->get_logger(), 
                "[NavigateToSubregionCenter] Sending goal: (x=%.2f, y=%.2f, theta=%.2f)",
                current_goal_.pose.position.x,
                current_goal_.pose.position.y,
                0.0); // 简化打印，实际可按需解析 orientation

    // 3. 封装 Goal 消息并异步发送
    NavAction::Goal goal_msg;
    goal_msg.pose = current_goal_;

    // 使用默认的 SendGoalOptions，仅通过 Future 轮询状态
    auto send_goal_options = NavActionClient::SendGoalOptions();
    goal_handle_future_ = client_->async_send_goal(goal_msg, send_goal_options);

    // 进入等待接受状态，返回 RUNNING 交由框架在下一 tick 调用 onRunning()
    state_ = NavState::WAITING_FOR_ACCEPTANCE;
    active_goal_handle_ = nullptr;
    return BT::NodeStatus::RUNNING;
}

BT::NodeStatus NavigateToSubregionCenter::onRunning()
{
    // 状态机流转：全程使用 wait_for(0s) 进行非阻塞轮询
    switch (state_) {
        case NavState::WAITING_FOR_ACCEPTANCE: {
            auto fut_status = goal_handle_future_.wait_for(0s);
            if (fut_status == std::future_status::ready) {
                try {
                    auto handle = goal_handle_future_.get();
                    if (!handle) {
                        RCLCPP_ERROR(node_->get_logger(), 
                                     "[NavigateToSubregionCenter] Goal rejected by Nav2 server.");
                        return BT::NodeStatus::FAILURE;
                    }

                    // 服务器已接受 Goal，保存 Handle 并获取结果 Future
                    active_goal_handle_ = handle;
                    result_future_ = active_goal_handle_->async_get_result();
                    
                    // 流转至结果等待状态
                    state_ = NavState::WAITING_FOR_RESULT;
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(node_->get_logger(), 
                                 "[NavigateToSubregionCenter] Goal handle fetch error: %s", e.what());
                    return BT::NodeStatus::FAILURE;
                }
            }
            return BT::NodeStatus::RUNNING;
        }

        case NavState::WAITING_FOR_RESULT: {
            auto fut_status = result_future_.wait_for(0s);
            if (fut_status == std::future_status::ready) {
                try {
                    auto wrapped_result = result_future_.get();
                    
                    switch (wrapped_result.code) {
                        case rclcpp_action::ResultCode::SUCCEEDED:
                            RCLCPP_INFO(node_->get_logger(), 
                                        "[NavigateToSubregionCenter] Navigation succeeded.");
                            return BT::NodeStatus::SUCCESS;

                        case rclcpp_action::ResultCode::ABORTED:
                            RCLCPP_WARN(node_->get_logger(), 
                                        "[NavigateToSubregionCenter] Navigation aborted by server.");
                            return BT::NodeStatus::FAILURE;

                        case rclcpp_action::ResultCode::CANCELED:
                            RCLCPP_INFO(node_->get_logger(), 
                                        "[NavigateToSubregionCenter] Navigation canceled (likely by Halt).");
                            return BT::NodeStatus::FAILURE; // 可根据业务需求改为 SUCCESS

                        default:
                            return BT::NodeStatus::FAILURE;
                    }
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(node_->get_logger(), 
                                 "[NavigateToSubregionCenter] Result fetch error: %s", e.what());
                    return BT::NodeStatus::FAILURE;
                }
            }
            return BT::NodeStatus::RUNNING;
        }

        default:
            RCLCPP_ERROR(node_->get_logger(), "[NavigateToSubregionCenter] Invalid internal state.");
            return BT::NodeStatus::FAILURE;
    }
}

void NavigateToSubregionCenter::onHalted()
{
    // 当上层节点（如 Sequence/ReactiveSequence 失败、Fallback 切换分支）触发中断时调用
    RCLCPP_INFO(node_->get_logger(), 
                "[NavigateToSubregionCenter] Halt triggered. Requesting goal cancellation...");

    // 如果 Goal 尚未发送完成或正在执行，主动取消
    if (active_goal_handle_ && client_ && client_->action_server_is_ready()) {
        try {
            client_->async_cancel_goal(active_goal_handle_);
        } catch (const std::exception& e) {
            RCLCPP_WARN(node_->get_logger(), 
                        "[NavigateToSubregionCenter] Cancel request failed: %s", e.what());
        }
    }

    // 清理状态，防止下次 tick 或重新激活时状态残留
    state_ = NavState::IDLE;
    active_goal_handle_ = nullptr;
}