clc
clear all
close all

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);


x = [80 224 432 704 1040 1440];

K1 = [ 3.00 	3.29 	3.45 	3.54	3.78	4.0] + 1;
K3 = [ 23.02 	27.31 	29.80 	31.77 	33.27	34.54] + 1;
K5 = [ 35.44 	41.88 	45.94 	48.76 	50.93	52.85] + 1;
R1 = [ 4.00 	4.21 	4.45 	4.54 	4.78	5.0] + 1;
R3 = [ 17.01 	16.68 	18.24 	20.99 	21.35	22.3] + 1;
R5 = [ 27.52 	30.82 	33.58 	34.14 	35.99	36.98] + 1;



plot(x,K3,'--+','LineWidth',1);
hold on
plot(x,K5,'--*','LineWidth',1);
hold on

plot(x,R3,'-x','LineWidth',1);
hold on
plot(x,R5,'-s','LineWidth',1);
hold on



grid on

set(gca,'YLim',[0 56]);
set(gca,'YTick',[0:5:56]);

set(gca,'XLim',[0 1440]);
set(gca,'XTick',[0 80 224 432 704 1040 1440]);


legend('KeyFlow: hop count=3','KeyFlow: hop count=5','KeySet: hop count=3','KeySet: hop count=5','Location','northwest');
% title('Hop Count with Bits') 
xlabel('The number of switches');
ylabel('Forwarding label length (bits)');