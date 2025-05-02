clc
clear all
close all

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);


x = [80 224 432 704 1040];

K1 = [ 3.00 	3.21 	3.45 	3.54	3.78  ] ;
K3 = [ 23.02 	27.31 	29.80 	31.77 	33.27 ] ;
K5 = [ 35.44 	41.88 	45.94 	48.76 	50.93 ] ;
R1 = [ 5.00 	5.21 	5.45 	5.54 	5.78  ] ;
R3 = [ 17.01 	20.68 	23.24 	24.99 	26.35 ] ;
R5 = [ 27.52 	31.82 	35.58 	38.14 	39.99 ] ;
S1 = [ 5.00 	5.21 	5.45 	5.54 	5.78  ] ; 
S3 = [ 16.98 	20.68 	23.22 	24.90 	26.21 ] ;
S5 = [ 25.28 	30.44 	34.41 	36.89 	38.60 ] ; 


plot(x,K3,'--+','LineWidth',1);
hold on
plot(x,K5,'--*','LineWidth',1);
hold on

plot(x,R3,'-x','LineWidth',1);
hold on
plot(x,R5,'-s','LineWidth',1);
hold on

hold on
plot(x,S3,'-^','LineWidth',1);
hold on
plot(x,S5,'-h','LineWidth',1);
hold on

grid on

set(gca,'YLim',[0 55]);
set(gca,'YTick',[0:5:55]);

set(gca,'XLim',[0 1040]);
set(gca,'XTick',[0 80 224 432 704 1040]);


legend('KeyFlow: hop count=3','KeyFlow: hop count=5','Ours: Random, hop count=3','Ours: Random, hop count=5','Ours: Special, hop count=3','Ours: Special, hop count=5','Location','northwest');
% title('Hop Count with Bits') 
xlabel('The number of switches');
ylabel('The length of X (bits)');