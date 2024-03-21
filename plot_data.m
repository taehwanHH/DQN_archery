% .mat 파일 로드
data = load('data.mat');

% 로드된 데이터를 이용해 plot 생성

window_size = 50;
avg_total_res = double(data.sim_res)/100;

moving_avg = movmean(avg_total_res,window_size);


plot(moving_avg);
