# Distributions

`the possible values a variable can take and how frequently they occur`

* P(Y=y) or P(y) -> the probability function
    * Y -> the actual outcome of an event
    * y -> one of the possible outcomes

* some terms
    * mean = average value = μ(mu-mü-)
    * variance = how spread out the data is = σ^2 -sigma squared-
    * population data = all data
    * sample data = some data
        * sample mean = x̄-x bar-
        * sample variance = s^2
    * standard deviation = positive square root of variance = σ -sigma-
        * for samples its "s"

* Mean and Variance Formula
    * σ^2 = E(Y^2) - μ^2

<br>

# Types of Probability Distributions

* Discrete Distributions -> finite numbers of outcomes
    * Uniform Dist. -> when all outcomes are equally likely (like flipping a coin or rolling a die)
    * Bernoulli Dist. -> TWO possible outcomes like "true and false"
    * Binomial Dist. -> similar experiment several times in a row, each iteration has two possible outcomes
    * Poisson D. -> test out how unusual an event frequency is for given interval

* Continuous Dist. -> infinitely many outcomes
    * Normal D. -> often observed in nature
    * Student's-T D. -> a small sample approximation of a Normal D. = normal d. with a limited data or time
    * Chi-Squared D. -> asymmetric, only consists of non-negative values
        * mostly used in hypothesis testing
    * Exponential D. -> events that are rapidly changing early on
    * Logistic D. -> useful in forecast analysis

## Discrete Distributions and its Characteristics

* finitely many distinct outcomes
* P(Y<=y) = P[Y < (y+1)]

### The Uniform Distribution: U(a,b)

* X ~ U(a,b)
* all outcomes have equal probability (like rolling a die or flipping a coin)

### Bernoulli Distribution: Bern(p)

* 1 trial, 2 possible outcomes
* 1 outcome has the possibility "p", while the other has "1-p"
* variance = σ^2 = p(1-p)

### Binomial Distribution: B(n,p)

* n trial, 2 possible outcomes for each trial
* P(desired outcome) = p
* p(alternative outcome) = 1-p
* p(y) = Combination(n,y) * p^y * (1-p)^(n-y)
* expected value = E(x) = x_0 * p(x_0) + ... + x_n * p(x_n)
    * Y ~ B(n,p)   -> E(Y) = n*p
* variance = σ^2 = E(y^2) - [E(y)]^2 = n*p*(1-p)

### Poisson Distribution: Po(λ)

* frequency with which an event occurs
* P(Y) = (λ^y * e^-y)/y!

## Characteristics of Continuous Distributions

* their sample space is infinite
    * so cannot record the frequency of each distinct value
    * so P(X) = 0, and P(x>X) = P(x>=X)
    * e.g P(x<6) = P(x<=6), and P(x=6) = 0

### Normal Distribution: N(μ, σ^2)

* E(x) = μ
* Var(X) = σ^2 = E(X^2) - [E(X)]^2
* Bell-shaped graph

### Standard Normal Distribution: Z(0, 1)

* E(X) = μ = 0
* Var(X) = 1
* Y -> Z
    * Z = (Y-μ) / σ

### The Students'T Distribution: T(k)

* if k>2
    * E(Y) = μ
    * Var(Y) = (S^2 *k) / (k-2)

### Chi-Squared Distribution: X^2(k)

* asymmetric
* have a table of known values: N,T
* E(X) = k, Var(X)=2k

### Exponential Distribution: Exp(λ)

* E(Y) = 1/λ
* Var(Y) = 1/λ^2
* No table

### Logistic Distribution: Logistic(μ, S)

* E(y) = μ
* Var(Y) = (S^2 * pi^2) / 3