In this example, we show a car running through the corridor to the goal.
```
| car |
|     |
|     |
|     |
|     -----------
|            goal
-----------------
```
We learn the dynamics of the car with training data that the car always runs within the corridor (never hits the wall), and then optimize the path of the car with the learned model and a score function regularization.