---
layout: post
title:  "Goal-Conditional Policy Gradients"
---

##### _16-745: Optimal Control and Reinforcement Learning_

###### Author: Ryan Brigden (rbrigden at cmu dot edu)
###### Project Code: [github.com/rbrigden](https://github.com/rbrigden/multi-goal)


### Introduction

A current trend in continuous control for robotics is to develop model-agnostic learning algorithms that can be applied to a variety of robotic systems and tasks. One popular method takes a reinforcement learning approach to learn parametric policies by gradient descent. Although these policy gradient (PG) methods have demonstrated the ability to learn robust policies for complex systems and tasks without the need of a model, they suffer from very high sample complexity. The natural reward structure for many problems of interest is sparse, which further degrades the learning ability of PG methods both in sample complexity and stability. I am interested in methods to learn policies that can accomplish a continuous set of tasks parameterized by a goal. Each of these tasks have individually sparse reward structure, but when considered in aggregate, induce a dense reward structure. This is feasible when the goal structure is smooth and the ability to achieve one goal is highly indicative of how to achieve those nearby it.

In this report I describe prior approaches to learning such goal-conditional policies and suggest an improvement in efficiency and stability for learning goal-conditional policies.

### Preliminaries

#### Sparse Rewards and Problems with Reward Specification

Ultimately we wish to tie the maximization of expected return across trajectories to achieving an exact desired state of the system. This is done by specifying a sparse reward structure for a task, which only places positive reward on such desired states. Unfortunately this requirement only compounds the problems for current PG methods, which often perform gradient descent based on the return and other information gathered over partial trajectories. If in the case of sparse reward structure the return for the majority of trajectories is zero, the gradients exhibit high variance. On-policy methods, which cannot refer to past experiences after updating, are particularly prone to catastrophic failure after a bad update.

A common adjustment made in practice is to apply reward shaping, but this is a generally unprincipled approach as it is unlikely to share the same set of optimal policies as the original sparse reward task and requires an amount of domain knowledge that scales in the complexity of the task.

The ideal approach would let us use a variant of PG methods to learn optimal policies on sparse reward tasks without incurring the aforementioned pitfalls.


#### Goal-Conditional Policies


Despite the variable complexity of robotic systems of interest, it is often simple to decompose the state of the system into a smooth goal representation of which the original goals of interest are a subset. We introduce a function $$h : S \rightarrow G$$ that maps each state into a goal representation. For example, if we wish to learn a policy for a humanoid to place its end effector on a surface, we can construct $$h$$ to map from the joint angles to the position of the end effector. In this case we are interested in learning a policy that can achieve a set of goals $$G' \subset \{h(s) : s \in S \}$$ (the states in which the end effector is on the surface). Given continuous states and that $$h$$ is a smooth function, what is learned about achieving goals not in $$G'$$ can tell us about how to achieve the goals in $$G'$$.

In order to leverage this underlying goal structure, we introduce the notion of a goal-conditional policy
$$ \pi(a | s, g)$$ that prescribes an action for each state, goal tuple. In the considered case of continuous states and goals, the goal-conditional policy is a parametric function
$$\pi_\theta(a | s, g)$$.

To put this all into the reinforcement learning framework, we can nicely place the original sparse reward problem in the context of our new goal representation. In the humanoid example, we would typically designate a fixed positive reward for reaching the states in which the end effector is on the surface and zero reward for all other states. With respect to our goal formulation this reward specification would be $$R(s) = k \cdot 1(h(s) \in G')$$, where $k$ is the fixed positive reward. We can then consider the goal-conditional reward function, which we will need to learn goal-conditional policies. The general goal-conditional reward function is $$R(s, g) = k \cdot 1(h(s) = g)$$.

Our ultimate goal is to learn a goal-conditional policy that maximizes the expected return over both the distribution of state trajectories and the distribution of goals. By setting up this more general problem, we induce a more dense reward function over the state space that maintains the same sparse reward specification per goal.

### Methods

#### Hindsight Policy Gradients (HPG)


We wish to learn a goal-conditional policy using a PG method such as REINFORCE. The goal-conditional policy gradient is a joint expectation of the return over trajectories and goals, given by

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau | \theta, g') \sum_g P(g) \sum_t \nabla_\theta \log \pi(a_t | s_t, \theta, g) G(s_t, g) $$


Unfortunately there is a not a neat way to sample trajectories generated under the goal distribution. Referring again to the simple example of a humanoid positioning its end effector on a surface, it is not possible to change the position of the target surface as we wish. An idea introduced in Andrychowicz et. al[^2] is to consider that the set of trajectories under the distribution of original goals,
$$\tau \sim \pi(a | s, g'),\,\,\, g' \sim G'$$, will contain states in which goals other than $G$ are achieved. This insight lets us sample experience and in "hindsight" consider that experience in the context of achieving a different goal.

Andrychowicz et. al[^2] use this line of reasoning to develop an off-policy algorithm based on deep deterministic policy gradients (DDPG) that learns a deterministic goal-conditional policy. Because DDPG is off-policy, it is relatively easy to adjust the sampled trajectories to incorporate the alternative goals. DDPG, however, is notoriously brittle and requires a large amount of hyperparameter tuning to learn reasonable policies on new tasks. Meanwhile, policy gradient methods have been shown to be more robust and less sensitive to hyperparameters than DDPG and similar off-policy methods.

Rauber et. al[^1] derives several policy-gradient estimators for goal-conditioned policy learning. The fundamental difficulty of applying hindsight to policy gradient methods is that the gradient estimator relies on the the trajectories being sampled under the policy pursuing a goal in $$G'$$. Therefore we can can re-imagine those same trajectories under an alternative goal naively. To get around this problem, Rauber et. al[^1] introduce an estimator that uses importance sampling to re-weight each step in the trajectory by the probability of the trajectory up until that point. This estimator is given by:

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau | \theta, g') \sum_g P(g) \sum_t \nabla_\theta \log \pi(a_t | s_t, \theta, g) \sum_{t' = t + 1}^H \prod_{k = 1}^t \frac{\pi(a_k | s_k, g)}{\pi(a_k | s_k, g')} A(a_{t'}, s_{t'}, g) $$

It is reasonable to believe that the policy under the original goal $$g'$$ is a decent proposal distribution for estimating the likelyhood of the trajectory under alternative goals $$g$$ because $$g$$ is near $$g'$$ and the goal structure is smooth. That finally thought is purely supposition and would be interesting to investigate more formally.

#### Advantage Actor-Critic HPG


REINFORCE is well known as a suboptimal PG algorithm and the current application of the HPG estimators to the REINFORCE algorithm as in Rauber et. al[^1] could limit their potential. Here I propose an adaptation of the Advantage Actor-Critic (A2C) algorithm[^3] that incorporates the HPG estimator, A2C-HPG. The REINFORCE algorithm using the HPG estimator to learn goal-conditional policies will henceforth be referred to as REINFORCE-HPG

The original A2C algorithm improves the learning efficiency and stability over REINFORCE in two ways.

1. Introduce value function approximator as a baseline for reducing variance of the Monte-Carlo return estimate. The baseline subtracted return is equivalent to the advantange $$A(a, s, g)$$.

2. Collect partial instead of full trajectories across $$N$$ workers for each update. Use the value function approximator to bootstrap the value of the last state in the partial trajectory if a terminal state is not reached.

Incorporating hindsight into A2C requires a reworking of the original algorithm. The main difference is that for each the $$N$$ partial trajectories $$\tau_i$$ sampled under original goal $$g'$$, a state is randomly selected from the trajectory and designated as an active goal $$g_i$$. We then consider a sub-trajectory of the original trajectory for each $$g_i$$. In the standard case in which an episode does not terminate for the entire trajectory
$$\tau_i$$, the sub-trajectory is from the beginning of the trajectory
$$\tau_i$$ up until the active goal $$g_i$$. In other cases when there is one or more episode terminations in $$\tau_i$$, we consider the sub-trajectory from the last episode termination before $$g_i$$ up until $$g_i$$ is reached.

Unlike REINFORCE-HPG, A2C-HPG incorporates a value function approximator $$V_\phi(s, g)$$ parameterized by $$\phi$$. The value function is also goal-conditional and we must fit it over the goal distribution as well. The loss of $$V_\phi(s, g)$$ for a set of trajectories $$T$$

$$ L(\phi) = \frac{1}{|\tau_i| |T|} \sum_{g, \tau \in T} \sum_{s \in \tau} \,\, (V_\phi(s, g) - G(s, g))^2$$

Advantages are also goal-conditional and are computed along the active goal trajectories with $$A_\phi(a_t, s_t, g_i) = G(s_t, g_i) - V_\phi(s_t, g_i)$$


### Experiments and Results

![seek](/assets/seek.gif)


The following experiment is done with a simple particle environment to verify the correctness of HPG-A2C and demonstrate its ability to learn effectively in sparse-reward environments over A2C.

The task is a simple 2D continuous "seeking" challenge in which an agent has to navigate to a landmark as quickly as possible. In each episode, the landmark and agent are placed randomly on the map. The episode ends when the agent reaches the landmark or runs out of time trying to reach it.

The task has two variations: sparse and shaped. The sparse variant provides the agent with fixed positive reward if it reaches the landmark and provides a negative constant reward at every other timestep to encourage the agent to reach the goal as fast as possible. The shaped variant provides the agent with a reward proportional to negative squared distance between the landmark and agent.


![shaped_env_stats](/assets/shaped.svg)

In the above training curve, it is evident that A2C has no problem attaining the near maximum possible return (0)
in the shaped reward variant of Seek.


![sparse_env_stats](/assets/sparse.svg)

In the sparse reward environment, regular A2C initially reaches some of the targets but then its performance
drops catastrophically and cannot recover. A2C-HPG on the other hand can solve this task. Note that the return
corresponding to optimal performance on the task differs between the sparse and shaped variants.


### Conclusion

We have motivated the reason for learning policies that learn on sparse reward structure. Primarily we have focused on an approach that
solves a harder problem (learning a goal-conditional policy) in order to get around some of the pitfalls of using PG methods on sparse reward
tasks. These simple experiments along with prior works show that there is potential for these approaches to accelerate and stabilize PG training
on certain continuous control tasks.

However, there a number of questions that bring up the limits and applications of the goal-conditional policy approach.

1. Can we predict the level of domain knowledge needed to reason about goal structure that is unique to any other task?
2. How or to what scale does the exploitation of goal structure constrain the application of this approach to other tasks?
3. How well do the policies generalize to other parts of the goal space that haven't been experienced yet?

### References


[^1]: Rauber et. al, Hindsight Policy Gradients https://arxiv.org/abs/1711.06006
[^2]: Andrychowicz et. al, Hindsight Experience Replay tps://arxiv.org/abs/1707.01495
[^3]: Wu et. al, Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation https://arxiv.org/abs/1708.05144
