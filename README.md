#  Learning a Multi-Robot Task Allocation Strategy with Considerations for Dynamic Ocean Currents
This work was completed as a class project for ROB 537: Learning-Based Control.

## Introduction
Climate scientists are continually working to understand the drivers and impacts of climate change on Earth’s oceans.
Of particular interest to these researchers are areas of the ocean floor near polar ice shelves, where warming waters are causing massive portions of ice to break off into the ocean. 
Multi-robot teams are well-suited to enter into  this dangerous environment to collect data for researchers because of their efficiency for completing tasks in parallel and redundancy in numbers. 
Due to the remote nature of the ocean floor, it is important that these teams are capable of autonomously allocating tasks to develop long-term mission schedules.
However, few existing task allocation methods for multi-robot teams account for the dynamics introduced by time varying ocean currents.

This repository contains a reinforcement learning approach for teaching a dispatcher agent to allocate a team of underwater robots to tasks in the presence of time varying ocean currents. 
We show that this method is capable of outperforming a greedy task assignment baseline approach after a short training period.
With additional development, this work may be extended to provide a reliable task scheduling heuristic for a team of robots in the presence of complex time-varying flow.

## Problem Description
We consider a set of $m$ robots and %n$ tasks, where each task $i$ has associated robot count and execuion time requirements.
We model this problem as a graph with nodes corresponding to tasks and edges representing the time-varying travel time between tasks.
To capture dynamic ocean currents, we model a randomly-initialized 2D vector that is projected on all edge weights.
This vector is updated at every time step, thus updating transition times.
Finally, the objective of this problem is to minimize the makespan, or the finishing time of the last completed task.

TODO: Insert problem image

## Deep Q-Network and Training
As the state space for this problem is extensive and continuous, we determined that a DQN would provide the best method for estimating Q-values for making task assignments given the current state of the environment.
To stably and reliably train a DQN, a target network and replay buffer are used.
The target network is a previous copy of the training network that is updated to match the training network at a set frequency.
This allows for loss to be calculated using the difference in the value predicted by the training network and the discounted value estimated by the target network. 
The loss is then backpropagated to update the weights of the training network.

The replay buffer keeps track of previous states, actions, rewards, and next states to ensure the network does not forget the previous information that contributed to its current configuration. 
During training, batches are randomly sampled from the entire population stored in the replay buffer.

TODO: Insert training diagram

