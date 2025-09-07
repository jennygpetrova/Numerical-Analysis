This report demonstrates several methods of polynomial interpolation. For a given set of $n+1$ distinct nodes $(x_0, y_0)$, \ldots, $(x_n, y_n)$, we want to construct a degree-$n$ polynomial $p_n(x) \in \mathbb{P}_n$ such that
\
$p_n(x) = a_0 + a_1 x + a_2 x^2 + \ldots + a_nx^n$, 
\
where $p_n(x_0) = y_0$, \ldots , $p_n(x_n) = y_n$. 

We construct routines to perform interpolation by the Barycentric Form 1 and Barycentric Form 2 of the Lagrange interpolating polynomial, and calculate the Newton divided differences, using the adapted Horner's rule to evaluate the Newton interpolating polynomial. The stability, accuracy, and computational complexity of each method is assessed.
