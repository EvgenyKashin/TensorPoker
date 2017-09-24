# TensorPoker
Neural networks based poker bots

This project consists of three main parts:
1. DQN based poker bot
2. A3C based poker bot
3. Cards embeddings

### DQN based poker bot
The most detailed solution - modular, customizable, commented and tested.

### A3C based poker bot
Solution placed in one jupyter notebook (todo: make it modular). It more powerfull and efficency.
It uses threading. Need more tests. 

### Cards embeddings
Embeddings for cards in a hand. Conv net was used. Trainig by predicting winrate from cards.

All parts use TensorFlow and TensorBoard, witch visualize losses, perfomance and validation metrics.

Tested in PyPokerEngine (Texas Holdem, no limit).