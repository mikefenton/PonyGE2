function get_plot(algorithm, timestamp)

tit = algorithm;
path = pwd;
target = strcat(strcat(strcat('/../network_plots/', timestamp),'/Heatmaps/output/'),algorithm);
dir = strcat(path,target);

fig = figure('visible','off');
colormap(fig, 'jet');
set(fig, 'Position', [100, 100, 1500, 895]);

u = load(strcat(dir, '.mat'));
vals = double(u.mat)*100;
h = bar3(vals);

shading interp;
for i = 1:length(h)
     zdata = get(h(i),'Zdata');
     set(h(i),'Cdata',zdata);
     set(h,'EdgeColor','k');
end

grid('off')
title(tit, 'FontSize', 30);
xlabel({'Centre UEs          Edge UEs'}, 'FontSize', 15);
%ylabel({'ABS             non-ABS';'         Subframes'}, 'FontSize', 15);
ylabel({'Subframes'}, 'FontSize', 15);

xlab = get(gca,'xlabel');
ylab = get(gca,'ylabel');
tit_handle = get(gca,'title');
set(ylab, 'Position', [0 2.8 0.0]);
set(xlab, 'Position', [0 -0.9 18.5]);
set(tit_handle, 'Position', [0 1 480]);
set(gca,'FontSize',15)

set(xlab,'Rotation',-67);
set(ylab,'Rotation',18);

set(gca,'XTickLabel',{});
set(gca,'YTickLabel',{});
view(250, 60);
zlim([0,100])
caxis([0,100]);

set(gcf, 'PaperPosition', [-0.5 -0.1 6.2 5.3]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [4.8 5.3]);
print(fig,dir,'-dpdf')
