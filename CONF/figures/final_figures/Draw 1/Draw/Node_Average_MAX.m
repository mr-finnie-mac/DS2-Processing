clc
clear all
close all

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);

x = [80 224 432 704 1040 1440];

KeyFlowA = [35.06582338578405 41.744916312928815 45.88119022497533 48.72705049585202 50.89248545721971 52.84235360821495];      % KeyFlow average bits
OursRA = [27.20835594139989 30.47301018372566 33.02080000680712 34.69013042958772 35.366008984059256 36.97655211516049];   % Our method average bits


KeyFlowM = [40 47 51 54 57 59];      % KeyFlow MAX bits
OursRM = [41 48 52 55 58 60];   % Our method MAX bits

KeyFlowA = KeyFlowA + 1;
OursRA = OursRA + 1;

KeyFlowM = KeyFlowM +1;
OursRM = OursRM +1;

plot(x,KeyFlowM,'--+','LineWidth',1); 
hold on

plot(x,KeyFlowA,'--s','LineWidth',1); 
hold on

plot(x,OursRM,'-*','LineWidth',1); 
hold on

plot(x,OursRA,'->','LineWidth',1); 
hold on

grid on


set(gca,'YLim',[20 62]);
set(gca,'YTick',[20:10:62]);

set(gca,'XLim',[0 1500]);
set(gca,'XTick',[0 80 224 432 704 1040 1440]);

legend('KeyFlow: MAX','KeyFlow: Average','KeySet: MAX','KeySet: Average','Location','northwest');
%title('Special Assignment') 
xlabel('The number of switches');
ylabel('Forwarding label length (bits)');