@doc doc"""
#### Objet
Cette fonction calcule une solution approchée du problème

```math
\min_{||s||< \Delta}  q(s) = s^{t} g + \frac{1}{2} s^{t}Hs
```

par l'algorithme du gradient conjugué tronqué

#### Syntaxe
```julia
s = Gradient_Conjugue_Tronque(g,H,option)
```

#### Entrées :   
   - g : (Array{Float,1}) un vecteur de ``\mathbb{R}^n``
   - H : (Array{Float,2}) une matrice symétrique de ``\mathbb{R}^{n\times n}``
   - options          : (Array{Float,1})
      - delta    : le rayon de la région de confiance
      - max_iter : le nombre maximal d'iterations
      - tol      : la tolérance pour la condition d'arrêt sur le gradient

#### Sorties:
   - s : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \Delta} q(s)``

#### Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(g,H,options)

    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        delta = 2
        max_iter = 100
        tol = 1e-6
    else
        delta = options[1]
        max_iter = options[2]
        tol = options[3]
    end
    #Initialisation
    n = length(g)
    s = zeros(n)
    p = -g

    function q(sk, sigma, pk)
        return g'*(sk+sigma*pk)+0.5*(sk+sigma*pk)'*H*(sk+sigma*pk)
    end
 
    j=0
     while (j<2*n) && (norm(g) > tol)
        # Mise a jour des variables
         k = p'*H*p
         a = norm(p)^2
         b = 2*p'*s
         c = norm(s)^2-delta^2
         sigma1 = (-b+(sqrt(b^2-4*a*c)))/(2*a)
         sigma2 = (-b-(sqrt(b^2-4*a*c)))/(2*a)

         if k<=0
            q1 = q(s, sigma1, p)
            q2 = q(s, sigma2, p)

             if q1<q2   # la racine de ∥sj + σpj∥ = ∆ pour laquelle q(sj + σpj) est la plus petite
                 s = s+sigma1*p
             else
                 s = s+sigma2*p
             end
             return s
         end

         alpha = g'*g/k
         if norm(s+alpha*p) >= delta
             if sigma1>=0   # la racine positive de ∥sj + σpj∥ = ∆
                 s = s+sigma1*p
             else
                 s = s+sigma2*p
             end
             return s
         end
         #Mise a jour des variables
         s = s+alpha*p
         g_old = g
         g = g+alpha*H*p
         beta = g'*g/(g_old'*g_old)
         p = -g+beta*p
         j = j+1
     end
 
    return s

end
