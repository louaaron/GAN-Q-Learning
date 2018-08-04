This code implements the "GAN Q-Learning" algorithm found in https://arxiv.org/abs/1805.04874. 

## TODO 

- [x] Example architectures 

- [ ] Example results

### Modifications From Paper

* The published algorithm has a typo in it (in the form of the discriminator loss)

* Currently, there seems to be a bug which causes the discriminator to (eventually) perfectly discriminate against the generator (even before learning the actual distribution)