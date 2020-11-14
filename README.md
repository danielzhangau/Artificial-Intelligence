# Artificial-Intelligence
The course aims to introduce the foundational concepts and methods used in the field of artificial intelligence and provide students with skills to apply these techniques. Specifically the course aims to give students an overview of the following topics in artificial intelligence:
- searching for solutions to problems,
- reasoning and planning with certainty, 
- reasoning and planning under uncertainty,
- learning to act, and
- reasoning about other agents.

## Assignment 1 - LaserTank UCS/A*
*Best-first search does not estimate how close to goal the current state is, it estimates how close to goal each of the next states will be (from the current state) to influence the path selected.*

Uniform-cost search expands the least cost node (regardless of heuristic), and best-first search expands the least (cost + heuristic) node.

- f(n) is the cost function used to evaluate the potential nodes to expand
- g(n) is the cost of moving to a node n
- h(n) is the estimated cost that it will take to get to the final goal state from if we were to go to n
The f(n) used in uniform-cost search
> f(n) = g(n)  

The f(n) used in best-first search (A* is an example of best-first search)  
> f(n) = g(n) + h(n)

Each of these functions is evaluating the potential expansion nodes, not the current node when traversing the tree looking for an n that is a goal state  
## Assignment 2 - Robot arm
Configuration Space
## Assignment 3 - LaserTank MDP
some useful learning resources: https://www.youtube.com/watch?v=HEs1ZCvLH2s&ab_channel=stanfordonline Lecture 6,7,8
## Assignment 4 - LaserTank Q-Learning and SARSA
some useful learning resources: https://www.youtube.com/watch?v=OkGFJE_XDzI&t=417s&ab_channel=%E8%8E%AB%E7%83%A6Python 莫烦python

## What I have learned:
1.	Describe the core theoretical and conceptual frameworks, methods and practices which form the basis of artificial intelligence.
2.	Explain the properties and functions of a range of different artificial intelligence methods and to be able to connect a method to appropriate theoretical foundations.
3.	Effectively solve problems relating to artificial intelligence topics and applications discussed in class and in the literature.
4.	Implement techniques and methods from artificial intelligence using a high-level programming language.
5.	Effectively formulate real-world problems as problem representations solvable by existing techniques in artificial intelligence.
