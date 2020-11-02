% Author: Xiaoming Xue
% Email: xminghsueh@gmail.com
%           xming.hsueh@my.cityu.edu.hk
%
% ------------
% Description:
% ------------
% This file is the entry point for running the the AT-MFEA method
% on a suite of modified multi-tasking benchmark problems.
%
% ------------
% Reference:
% ------------
% X. Xue, K. Zhang, K. C. Tan, L. Feng, J. Wang, G. Chen, X. Zhao, L. Zhang, 
% and J. Yao, ¡°Affine Transformation Enhanced Multifactorial Optimization 
% for Heterogeneous Problems,¡± IEEE Transactions on Cybernetics, pp. 1¨C1, 2020.

%%
clc
clear all;
warning off;


pop = 100; % population size for multitasking
gen = 1000; % generation count
selection_pressure = 'elitist'; % choose either 'elitist' or 'roulette wheel'
p_il = 0; % probability of individual learning (BFGA quasi-Newton Algorithm) - local search (optional)
rmp = 0.3;
reps = 20; % reps > 1 to compute mean rmp values
num_func = 9; % the number of multitasking functions to be optimized

for fun_index = 1:num_func
     
    Tasks = benchmark_modified(fun_index);
    
    data_ATMFEA(fun_index) = ATMFEA(Tasks,pop,gen,rmp,selection_pressure,p_il,reps,fun_index);
    
end

%%
save('result_ATMFEA_p100_g1000.mat','data_ATMFEA');