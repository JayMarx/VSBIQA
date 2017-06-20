m = load('./live2_test/CORNIA.txt');
test_img = m(:, 1);
pre_score = m(:, 2);
mat = load('./dmos.mat');
sub_score = [ ];
for i = 1 : length(test_img)
    index = int32(test_img(i));
    sub_score = [sub_score; mat.dmos(index)];
end


fit = polyfit(pre_score, sub_score, 3);
x_min = min(pre_score);
x_max = max(pre_score);
x1 = x_min:x_max;
y1 = polyval(fit, x1);

plot(pre_score, sub_score, 'x', x1, y1, '-r');
legend('All distortion', 'Fit curve')
xlabel('Estimated Quality');
ylabel('DMOS');
set(gca,'FontSize', 18);
SROCC = corr(pre_score, sub_score, 'type', 'Spearman');
PLCC = corr(pre_score, sub_score, 'type', 'Pearson');
fprintf('PLCC: %f\n', PLCC);
fprintf('SROCC: %f\n', SROCC);