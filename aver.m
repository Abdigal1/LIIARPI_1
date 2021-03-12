close all;clear;clc;
k = 1;
while k<10
    disp(k)
    opt = input('Save(s), Skip(k) or Retry(r)\n', 's');
    disp(opt)
    if strcmp(opt, 's')
        disp('Guardando')
        
    elseif strcmp(opt, 'k')
        
    elseif strcmp(opt, 'r')
        k = k-1
    else
        disp('kgado')
    end
    k = k+1;
end
