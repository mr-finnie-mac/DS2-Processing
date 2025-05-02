clc
clear all
close all

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);

x = [80 224 432 704 1040];

y1 = [35.06 41.77 45.88 48.73 50.91];      % KeyFlow average bits
y2 = [27.21 31.73 35.54 38.11 39.97];   % Our method average bits
y3 = [40 47 51 54 57];    % KeyFlow MAX bits
y4 = [42 48 53 56 58];   %  Our method MAX bits



plot(x,y3,'--o','LineWidth',1); %,'LineWidth',2);
hold on

plot(x,y1,'--^','LineWidth',1); %,'LineWidth',2);
hold on

plot(x,y4,'-d','LineWidth',1); %,'LineWidth',2);
hold on

plot(x,y2,'-x','LineWidth',1); %,'LineWidth',2);
hold on
grid on


set(gca,'YLim',[10 60]);
set(gca,'YTick',[10:10:60]);

set(gca,'XLim',[0 1040]);
set(gca,'XTick',[0 80 224 432 704 1040]);

legend('KeyFlow: Max','KeyFlow: Average','Ours: Max','Ours: Average','Location','southeast');
% title('Random Assignment') 
xlabel('The number of switches');
ylabel('The length of X (bits)');