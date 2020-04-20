# Product

## Overall Instruction:

* 考虑公司目标
* metric一定要和公司目标相关
* 考虑公司成长
* 考虑时间，长期短期指标
* 长期指标: find short term ones that can predict the long term one
* Both characteristics data (sex, gender, country, age) and behavioral data( browsing behavior)


## Take aways:

1. Improve engagement
    * Metric is related to the company mission. (FB: interaction; Quora: High level content; Airbnb: go to the given place)
    * Both characteristics data (sex, gender, country, age) and behavioral data( browsing behavior)
    * 中国人结果不错，30岁以上的不行，因此要看。。。
    * 针对不同的群体给出可能的建议
2. test-result good, why not launch?
    * cost(Human labor, Risks of bugs)
    * Look at effect size，是不是值得做
        * n很大的时候，很小的差别都能检测出来。这时候要判断做这个是不是有价值、有意义
3. missing value
    * 不能assume missing value和原有分布相同
        * 往往missing是用户选择不填的
    * 用ML预测missing，如果可以就替代；甚至可以drop，因为info已经包括在其他变量中
    * encode成-1，这对预测其他东西有作用；missing value是蕴含信息的
4. Drawback: supervised learning to predict fraud?
    * data skew严重，fraud概率低
    * fraud行为特征可能变化，有个人用A方式骗钱，下次可能用B方式
    * 用anomaly detection比较好
5. Different performance on IOS and Android
    * 看平台是不是proxy of other variables, such as age, education, gender
        * 用ML预测平台based on其他变量
        * 或者直接统计检测他们是不是相关
    * 平台问题
        * loading时间，bug，UI等
6. Long-term metric: increase retention rate or not
    * define retention metrics (% of users not unsubscribe within first 12 months)
    * ML 进行binary 预测，看是不是有short-term proxy可以预测long-term
        * encode历史数据，转换成0-1；例如retain的是1（1年没退订），否则是0
        * 用其他variable去预测，用最能预测的变量作为proxy
        * 如果是多个变量，直接用模型输出作为metric
    * 一般来讲，3个星期的data足够predict long term
7. Using demo data or behavioral data?
    * demo只能预测某类人会存在怎样的pattern
    * 有些情况下人们使用app不光是自己，也是为别人（送给别人礼物）
8. App performance
    * Ability to acquire new users
        * new sign-up with at least 1 message within X days （确保sign up是有效的，不是机器人）
    * Ability to retain current users/engagement
        * 选择希望用户产生的行为，例如多下单、多聊天等等
    * Ability to generate larger revenue per customer
9. New feature: new driver UI, to increase # of trips
    * novelty effect
        * 比较新注册用户和原用户的情况，因为新用户不存在novelty effect
    * 要选择不同但是comparable市场，因为司机之间相互影响（一个接单多了，一个会变少）
        * for each pair, randomly distribute them to new app or old app
10. What new features can be added
    * 从现有产品中，观察用户是否有其他需求
    * 对于可能的需求，转换成只需要（甚至不需要）点击就能完成的功能；不要让用户离开app
        * 例如what’s app， 可以分析文本，看用户是不是经常问别人有没有看到消息？问别人在哪里？
        * 开发新功能如显示已阅读、google map定位等
11. Do we really need to add the new feature?
    * feature会对公司产生价值么？
        * 是否会影响重要的metric，例如retention
        * 是否能从已有数据证明有这样的需求
    * cost是怎么样的，可能流失用户/增加成本等
    * need to find a proxy for demand of that feature in your current data 
        * 新feature 是不是simply the path of original function（例如FB，NLP后发现很多人会评论喜欢，这时候就可以加个喜欢的button）
    * 满足以上两点（demand+value），还需要test是不是能够达到，test通过了才能加
12. Drawbacks of running A/B testing on different market
    * Users in different markets will never be as similar as users in the same market, more noise
    * Unpredictable things might happen during the test just in one of the two markets (new competitors)
    * 如果非要做
        * 确保其他metric不变
        * 仅仅是测试会对其他user有影响的feature时，即网络效应时
13. Decrease in the metric 
    * break down 这个metric看看是不是可以分成几个小的点，例如A/B
    * 对于A和B，逐个分析
    * 针对不同demographic的人群进行分析（不同类别）
        * 男生下降，女生没降等
14. 30 experiments at the same time, one result significant, do we need to apply
    * 概率很小的事件重复30次，也有可能发生
    * 需要用原α值除以实验个数，作为新的threshold (0.05/30)
15. 100个feature，用户至少在一个其中一个上成为outlier的概率
    * 尝试造一个适合所有人的产品（只看平均值）是不合适的，因为高维空间内没有一个用户处于平均值
16. Cost of FP and FN
    * 高cost of FP：招聘，招到不好的candidate成本很大（面试过程和随机森林很像）
    * 高cost of FN：癌症detection
    * 解决办法和结合产品（分成三类）。对于预测高的人如何处理，低的人如何处理，中等的人如何处理
17. Without enough data
    * 问题
        * train/test直接分的话可能没法代替原分布
        * 数据不足够使模型converge
    * 解决方法（前提是数据还是有足够的能表示原有分布的信息，不然再怎么bootstrap也没用）
        * Cross validation
        * Bootstrap
18. Metrics btw % of users performing an action/mean time of their response
    * 不要用vanity metrics
    * %  of users performing an action 一般都比较好
        * 可以考虑outlier
        * 分母中可以考虑whole population；如果是第二种(mean of all response time < 24 hrs)，那些从来没有response的人就不会被考虑到，而事实上他们是重要的。一个人从10hrs变成no response，反而可能提高metric（原mean<10的话）
19. Clickbait ads identification
    * 两个feature：一个是CTR，一个是两周后的CTR（前者高，后者低）。cluster得到label
    * 加入label去做prediction
20. Right skewed data
    * mode&lt;median&lt;mean（mean会被outlier影响）
21. Increase ads revenue:
    * Increase click through rate: improve data and model
    * Increase number of times ads are shown: 用户下滑时可以看到更多东西；提供多页供用户看
 

## Key-point
1. 为什么要这么做
    - 证据表明我们缺少这个feature
    - 这个和公司目标直接相关
2. 总体市场：seasonality，market trend
3. 宏观情况：economy, policy
4. 产品：是不是有哪些特征没有纳入
5. 用户：一定要分组去看指标
7. What kind of data do you have
    - demographic 
    - behavioral











