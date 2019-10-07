# tucker-decomposition-ng-julia
The Newton--Grassmann method for the Tucker decomposition of a third order tensor.  For more details, see https://doi.org/10.1137/070688316 and Tensor approximation algorithm package (https://www.itn.liu.se/personal/berkant/algorithms?l=sv).

MATLABのコードとして公開されているNG(Newton--Grassmann)法をもとにJuliaでNG法を書きました．
TensorToolbox(https://github.com/lanaperisa/TensorToolbox.jl)を利用するため，事前にインストールして下さい．(`using Pkg; Pkg.add("TensorToolbox")`)

# 利用例

```
# demo.jl
using LinearAlgebra
using TensorToolbox
using PyPlot
using Random

include("orthogonal_complement.jl")
include("hooi.jl")
include("ng.jl")

Random.seed!(10)
A = rand(20,20,20)
Gh,Xh,Yh,Zh = hooi(A,10,10,10,50)
Xn,Yn,Zn,reln = ngtensor(A,Xh,Yh,Zh,20,1,2)

plot(reln)
yscale("log")
show()
```
