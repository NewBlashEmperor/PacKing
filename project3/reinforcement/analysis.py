# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """
    # 折扣系数，表示目的地距离带来的折扣
    # 其值越小，表示距离越远的状态对当前状态的影响越小
    answerDiscount = 0.1 # 如果我设置的很小，那么它会更倾向于最近的那个
    # 意外执行其他的行动的概率
    answerNoise = 0 
    # 生存奖励，其值越低，agent 越想冒险
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """
    answerDiscount = 0.1 # 这样设置就会倾向于1
    answerNoise = 0.1 # 这个是意外执行其他运动的概率，越大的话说明越容易发生意外的动作，因此就会远离cliff
    answerLivingReward = 0 # 活着没有危害，这样的话就不会去冒险
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """
    answerDiscount = 0.9 # 要喜欢远处的10，那么这个Discount不能太小
    answerNoise = 0 # 敢冒险走cliff的路，说明意外执行其他的动作的概率应该比较小
    answerLivingReward = -0.01 # 让他最终结束
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    answerDiscount = 0.9
    answerNoise = 0.1
    answerLivingReward = -0.01
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """
    answerDiscount = 1
    answerNoise = 0.5 # 如果意外动作的概率比较大，就可以往避开cliff的路走
    answerLivingReward = 100 # 
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
