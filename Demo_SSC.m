% test for Single view Subspace clustering algorithm
% References:
%%--------------------------------------------------------------------------
% References:
% [1] Qiang Guo, Tinghe Yan, 
% Efficient subspace clustering via variational schatten-p norm minimization, Submit draft, 2024.
% [2] Canyi Lu, Xi Peng, Yunchao Wei. 
% Low-rank tensor completion with a new tensor nuclear norm induced by invertible linear transforms. 
% IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
% [3] Paris Giampouras, René Vidal, Athanasios Rontogiannis, Benjamin D. Haeffele.
% A novel variational form of the Schatten-p quasi-norm. 
% The Advances in Neural Information Processing Systems, 2020.
% [4] Yuan Xie, Dacheng Tao, Wensheng Zhang, Yan Liu, Lei Zhang, Yanyun Qu.
% On unifying multi-view self-representations for clustering by tensor multi-rank minimization.
% International Journal of Computer Vision, 2018.
% version 1.0 -- 1/3/2024
%
%%--------------------------------------------------------------------------
% Written by Tinghe Yan
%%--------------------------------------------------------------------------

clear
close all

addpath('ClusteringMeasure', 'Funs');

 % Setting the number of algorithm executions
   num_runs = 10;
   
  % Loading data
    load('Yale.mat');
  fprintf('Testing %s...\n', 'Yale');
     views=3;
    
    for k=1:views
        eval(sprintf('X{%d} = double(X%d);', k, k));
    end
    %  Number of clusters
    cls_num = length(unique(gt));

     % Normalization
        Y = X;
        for iv=1:views
            [Y{iv}]=NormalizeData(X{iv});
        end   
        
     % Setting parameters    
        lambda1 = 1e-5; 
        lambda2 = 7e-6; 
        lambda3 = 1e-5; 
        p = 0.5; 
        r=30;
      % Testing on the num-th view
        num=3; %
      

   %  Running the proposed algorithm
        for kk = 1:num_runs
            time_start = tic;
            [C, S, Out] = solving_SSC(Y, num,r,cls_num, lambda1,lambda2,lambda3,p,gt);
            alg_name{1} = 'VSpNM-SSC';
            Out.time=toc(time_start);
            alg_cpu(kk) = Out.time;
            alg_NMI(kk) = Out.NMI;
            alg_AR(kk) = Out.AR;
            alg_ACC(kk) = Out.ACC;
            alg_recall(kk) = Out.recall;
            alg_precision(kk) = Out.precision;
            alg_fscore(kk) = Out.fscore; 
            alg_purity(kk) = Out.purity; 
       end

    %% Results report
    flag_report = 1;
    if flag_report
        fprintf('%6s\t%12s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\n',...\
            'Stats', 'Algs', 'CPU', 'NMI', 'AR', 'ACC', 'Recall', 'Pre', 'F-Score', 'Purity');
        fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
                'Mean', alg_name{1},mean(alg_cpu),mean(alg_NMI),mean(alg_AR),...\
                mean(alg_ACC),mean(alg_recall),mean(alg_precision),mean(alg_fscore),mean(alg_purity));
        fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
                'Std', alg_name{1},std(alg_cpu),std(alg_NMI),std(alg_AR),...\
                std(alg_ACC),std(alg_recall),std(alg_precision),std(alg_fscore),std(alg_purity));
    end
