#ifndef NAVIGATE_TO_SUBREGION_CENTER_HPP
#define NAVIGATE_TO_SUBREGION_CENTER_HPP

#include <memory>
#include <string>
#include <future>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <behaviortree_cpp/behavior_tree.h>

/**
 * @brief 异步导航至子区域中心的行为树节点
 * 纯基于 StatefulActionNode 实现，内部维护状态机避免重复发送 Goal，全程非阻塞轮询。
 */
class NavigateToSubregionCenter : public BT::StatefulActionNode
{
public:
    /**
     * @brief 构造函数
     * @param name 节点在 XML 中的名称
     * @param config BT.CPP 节点配置对象
     * @param node 共享的 ROS2 Node 指针，用于创建 Action Client 和日志
     */
    NavigateToSubregionCenter(const std::string& name,
                              const BT::NodeConfiguration& config,
                              rclcpp::Node::SharedPtr node);

    /// @brief 定义黑板端口
    static BT::PortsList providedPorts();

private:
    // BT.CPP v4 生命周期回调（严格非阻塞）
    BT::NodeStatus onStart() override;
    BT::NodeStatus onRunning() override;
    void onHalted() override; // 注：v4 标准虚函数名为 onHalted()，对应题目要求的 onHalt 机制

    /// @brief 内部状态机枚举
    enum class NavState { 
        IDLE,                 ///< 初始/重置状态
        WAITING_FOR_ACCEPTANCE, ///< Goal 已发送，等待服务器响应接受/拒绝
        WAITING_FOR_RESULT    ///< 服务器已接受，等待最终导航结果
    };

    rclcpp::Node::SharedPtr node_;
    using NavAction = nav2_msgs::action::NavigateToPose;
    using NavActionClient = rclcpp_action::Client<NavAction>;
    NavActionClient::SharedPtr client_;

    NavState state_ = NavState::IDLE;
    geometry_msgs::msg::PoseStamped current_goal_;

    // 异步通信 Future 对象
    std::shared_future<NavActionClient::GoalHandle::SharedPtr> goal_handle_future_;
    std::shared_future<NavActionClient::WrappedResult> result_future_;
    NavActionClient::GoalHandle::SharedPtr active_goal_handle_;
};

#endif // NAVIGATE_TO_SUBREGION_CENTER_HPP