# Overview of feature selection
- Select some subsets feature, in which one of them make performance of one of models is the best. Depend on the model or algorithms, there is one of some subset in proportion to model.

**1. Why need to use FS ?**
- The simple models are easier to interpret
- Shorted to traning
- Reduce overfitting
- Easier to deploy
- Reducce risk of variable error of collection and storage
- Avoid use variables with highly correlation
- Poor performance in some type of model: tree-base

**2. Methods**
- Filter methods
    - Basis
        - Constanst
        - Quasi - constanst
        - Duplicated
    - Statistics measures
        - Fisher score
        - Univariate methods
        - Mutual information
    - Correlation
- Wrapped methods
    - Step forward selection
    - Step backward selection
    - Exhaustive search
    - Feature suffling
- Embedded methods
    - Lasso
    - Decision tree derived importance
    - Regression coefficients
- Hybrid methods = wrapped + embedded methods

Check out the content pages.

```{tableofcontents}
```