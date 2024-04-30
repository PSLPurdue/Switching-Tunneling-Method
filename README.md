Originally based on:
+ Adaptive Memory Programming for Global Optimization (AMPGO).
    
+ added to lmfit by Renee Otten (2018)
    
+ based on the Python implementation of Andrea Gavana (see: http://infinity77.net/global_optimization/)
    
+ Implementation details can be found in this paper:http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf


Significantly modified using techniques from:
+ S. Gomez and C. Barron, “The exponential tunneling method,” Ser. Reportes Investig. IIMAS, vol. 1, no. 3, 1991.
+ A. V. Levy and Susana, “The Tunneling Method Applied to Global Optimization,” in Numerical Optimization, P. T. Boggs, R. H. Byrd, and R. B. Schnabel, Eds. New York: Society for Industrial & Applied Mathematics, 1985, pp. 213–244.
+ L. Castellanos and S. Gómez, “A new implementation of the tunneling methods for bound constrained global optimization,” Rep. Investig. IIMAS, vol. 10, no. 59, pp. 1–18, 2000.
+ J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed. 2006.
    
Modifications include:
+ Use of exponential tunneling function rather than tabu   
+ Tunneling zero search method implemented according to Gomez and Barron
+ Wolfe criteria for step size
+ Implementation of log barrier constraints with iterative reduction of mu and associated slack and tolerance factors
+ Augmented Lagrangian Method to enforce equality constraints
+ Iteration history plotting available
+ Switching Method to use primary and secondary function transformations

Many of the parameters (minimum step sizes, tolerances, slack, etc) are problem
dependent and must be tested/adjusted accordingly.
    
See paper:
+ Inverse Design of Bistable Composite Laminates with Switching Tunneling Method for Global Optimization Katherine S. Riley, Mark H. Jhon, Hortense Le Ferrand, Dan Wang, Andres F. Arrieta


This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.