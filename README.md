# Cross-Stitch Network using Jax:
Implementation of a Cross-Stitch Network on Jax

---
Misra, I., Shivastava, A., Gupta, A., Herbert,M. (2016). Cross-Stitch Networks for Multi-Task Learning. 

Misra et al combined the activation map of a layer of one network to another network using a  learnable parameters alpha. 

Overview:
--- 
Two FC networks with [60,20,10] stucture are trained on either MNIST or Fashion MNIST. 
The two networks are combined with 'alpha' to form a network of shape [120,40,10] and trained on a fraction of a combined dataset. 

Result is a combined network that can classify mnist and fashion mnist with similar accuracy to the networks on their own. 
