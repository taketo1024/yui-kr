# yui-kr (結)

A program that computes the reduced HOMFLY homology of knots.   
See: K. Nakagane, T. Sano *"Computations of HOMFLY homology"*, [arXiv:2111.00388](https://arxiv.org/abs/2111.00388).

## Setup

* Install [Rust](https://www.rust-lang.org/tools/install).
* Download [`yui-kr`](https://github.com/taketo1024/yui-kr.git).
* Build

```sh
$ cargo build -r
```

* Run

```sh
$ ./target/release/ykr <TARGET> [OPTIONS]
```

## Usage 

```
Usage: ykr <TARGET> [OPTIONS]

Arguments:
  <TARGET>  

Options:
  -b, --braid <BRAID>                
  -i, --int-type <INT_TYPE>          [default: i64] [possible values: i64, i128, bigint]
  -m, --mirror                       
  -f, --format <FORMAT>              [default: poincare] [possible values: poincare, poincare-tex, delta, table]
  -F, --force-compute                
  -c, --compute-mode <COMPUTE_MODE>  [default: default] [possible values: default, per-col, per-item]
  -l, --limit <LIMIT>                
  -p, --save-progress                
      --debug                        
```

## Examples 

```
$ ykr 3_1
q⁻²a² + t⁻²q²a² + t⁻³a⁴
```

```
$ ykr 5_1 -b '[1,1,1,1,1]'
q⁻⁴a⁴ + t⁻²a⁴ + t⁻³q⁻²a⁶ + t⁻⁴q⁴a⁴ + t⁻⁵q²a⁶
```

```
$ ykr 4_1 -f delta
Δ: 0
 j\i  -2  0  2 
 2    .   1  . 
 0    1   1  1 
 -2   .   1  .
 ```

```
$ ykr 11a_263 -f delta
Δ: 6
 j\i  -2  0  2 
 14   .   1  . 
 12   1   .  1 
 10   .   1  . 

Δ: 8
 j\i  -8  -6  -4  -2  0  2  4  6  8 
 12   .   1   3   3   5  3  3  1  . 
 10   1   4   4   9   6  9  4  4  1 
 8    1   1   4   3   6  3  4  1  1 
```

Note: If precomputed result for the specified target exist in the `results` directory, `ykr` will load the data and print the result without computation. Specify `-F` option if you want to force computation. 

## Computation Results

Computations are peformed for all prime knots with up to 11 crossings. The braid representatives of prime knots provided in [KnotInfo](https://knotinfo.math.indiana.edu) were used as the input data.

* [PDF](results.pdf) (Human friendly)
* [JSON](results/) (Machine friendly)

## License
`yui-kr` is released under the [MIT license](LICENSE).
