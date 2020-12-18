# **Multi-Arm Bandits for recommendations and A/B testing on Amazon ratings data set**


Multi Arm Bandits is used by many companies like Stitchfix, Netflix, Microsoft, and other big companies for recommendations. There tons of research going on on the Multi-arm Bandits and their application to real-time problems.

To understand more about the Multi Arm Bandit please go through this article on medium

[Multi-Arm Bandits for recommendations and A/B testing](https://abhishek-maheshwarappa.medium.com/multi-arm-bandits-for-recommendations-and-a-b-testing-on-amazon-ratings-data-set-9f802f2c4073)

Data can be downloaded from the 

https://www.kaggle.com/saurav9786/amazon-product-reviews

## Amazon Product Reviews

* userId : Every user identified with a unique id (First Column)

* productId : Every product identified with a unique id(Second Column)

* Rating : Rating of the corresponding product by the corresponding user(Third Column)

* timestamp : Time of the rating ( Fourth Column)


# **Result**

Format: ![Alt Text](https://github.com/abhi-gm/Multi-Armed-Bandits-for-Recommendations-and-A-B-testing/blob/main/assets/Results.JPG)

# **Conclusion**

* From above it clear that Multi-arm bandit is way better compared to A/B testing

* Among the Multi-arm bandit, Thompson sampling works the best

The Thompson Sampling results (brown) are the best of them all. This bandit performs better than the ε-Greedy bandit because it dynamically adjusts the rate at which it explores — rather than using a constant rate. In the beginning, it explores more often, but over time, it explores less often. As a result, this bandit quickly identifies the best product and exploits it more frequently after it has been found, leading to high performance in both the short- and long-term.


