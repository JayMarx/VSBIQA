m = importdata('./cross_experiment_csiq/csiq_all.txt');
n = importdata('./cross_experiment_csiq/csiq_all_result.txt');
sub_score = m.data;
pre_score = n.data;

fit = polyfit(pre_score, sub_score, 3);
x_min = min(pre_score);
x_max = max(pre_score);
x1 = x_min:x_max;
y1 = polyval(fit, x1);
plot(pre_score, sub_score, 'x', x1, y1, '-r');
xlabel('Estimated Quality');
ylabel('DMOS');

SROCC = corr(pre_score, sub_score, 'type', 'Spearman');
PLCC = corr(pre_score, sub_score, 'type', 'Pearson');
fprintf('PLCC: %f\n', PLCC);
fprintf('SROCC: %f\n', SROCC);