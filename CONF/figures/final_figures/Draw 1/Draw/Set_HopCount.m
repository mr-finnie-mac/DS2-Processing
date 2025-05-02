clc
clear all
close all

%Figure for 80 nodes

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);

Hop_count   = [5];
OursR0      = [35.3701171875];
plot(Hop_count,OursR0,'-.s','LineWidth',1);
hold on

Hop_count   = [5];
OursR1      = [32.42724609375];
plot(Hop_count,OursR1,'-.*','LineWidth',1);
hold on

Hop_count   = [3, 5];
OursR2      = [18.848958333333332, 27.856665826612904];
plot(Hop_count,OursR2,'-.p','LineWidth',1);
hold on

Hop_count   = [1, 3, 5];
OursR3      = [5.0, 16.39236111111111, 24.89720394736842];
plot(Hop_count,OursR3,'-.>','LineWidth',1);
hold on

Hop_count   = [1, 3, 5];
OursR4      = [3.0, 11.625, 19.029052734375];
plot(Hop_count,OursR4,'-.<','LineWidth',1);
hold on

grid on

set(gca,'XLim',[0 6]);
set(gca,'XTick',[1 2 3 4 5]);

legend('Set(n)','Set(n/2)','Set(n/4)','Set(n/8)','Set(n/16)','Location','northwest');
xlabel('The number of hops, 80 nodes');
ylabel('The length of X (bits)');

%Figure for 224 nodes

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);

Hop_count   = [5];
OursR0      = [41.83653846153846];
plot(Hop_count,OursR0,'-.s','LineWidth',1);
hold on

Hop_count   = [5];
OursR1      = [36.82919520547945];
plot(Hop_count,OursR1,'-.*','LineWidth',1);
hold on

Hop_count   = [5];
OursR2      = [31.98055028462998];
plot(Hop_count,OursR2,'-.p','LineWidth',1);
hold on

Hop_count   = [5];
OursR3      = [29.004378434065934];
plot(Hop_count,OursR3,'-.>','LineWidth',1);
hold on

Hop_count   = [1, 3, 5];
OursR4      = [3.2333333333333334, 15.516865079365079, 24.583658854166668];
plot(Hop_count,OursR4,'-.<','LineWidth',1);
hold on

grid on

set(gca,'XLim',[0 6]);
set(gca,'XTick',[1 2 3 4 5]);

legend('Set(n)','Set(n/2)','Set(n/4)','Set(n/8)','Set(n/16)','Location','northwest');
xlabel('The number of hops, 224 nodes');
ylabel('The length of X (bits)');

%Figure for 432 nodes

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);

Hop_count   = [5];
OursR0      = [45.81469298245614];
plot(Hop_count,OursR0,'-.s','LineWidth',1);
hold on

Hop_count   = [5];
OursR1      = [41.00636574074074];
plot(Hop_count,OursR1,'-.*','LineWidth',1);
hold on

Hop_count   = [5];
OursR2      = [36.136952457264954];
plot(Hop_count,OursR2,'-.p','LineWidth',1);
hold on

Hop_count   = [5];
OursR3      = [33.17562367303609];
plot(Hop_count,OursR3,'-.>','LineWidth',1);
hold on

Hop_count   = [1, 3, 5];
OursR4      = [3.4782608695652173, 17.883419689119172, 28.527541908021767];
plot(Hop_count,OursR4,'-.<','LineWidth',1);
hold on

grid on

set(gca,'XLim',[0 6]);
set(gca,'XTick',[1 2 3 4 5]);

legend('Set(n)','Set(n/2)','Set(n/4)','Set(n/8)','Set(n/16)','Location','northwest');
xlabel('The number of hops, 432 nodes');
ylabel('The length of X (bits)');

%Figure for 704 nodes

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);

Hop_count   = [5];
OursR0      = [48.380208333333336];
plot(Hop_count,OursR0,'-.s','LineWidth',1);
hold on

Hop_count   = [5];
OursR1      = [43.455729166666664];
plot(Hop_count,OursR1,'-.*','LineWidth',1);
hold on

Hop_count   = [5];
OursR2      = [38.98625300480769];
plot(Hop_count,OursR2,'-.p','LineWidth',1);
hold on

Hop_count   = [5];
OursR3      = [35.984309895833334];
plot(Hop_count,OursR3,'-.>','LineWidth',1);
hold on

Hop_count   = [1, 3, 5];
OursR4      = [3.727272727272727, 19.248456790123456, 30.797592956469167];
plot(Hop_count,OursR4,'-.<','LineWidth',1);
hold on

grid on

set(gca,'XLim',[0 6]);
set(gca,'XTick',[1 2 3 4 5]);

legend('Set(n)','Set(n/2)','Set(n/4)','Set(n/8)','Set(n/16)','Location','northwest');
xlabel('The number of hops, 704 nodes');
ylabel('The length of X (bits)');

%Figure for 1040 nodes

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);

Hop_count   = [5];
OursR0      = [51.395833333333336];
plot(Hop_count,OursR0,'-.s','LineWidth',1);
hold on

Hop_count   = [5];
OursR1      = [45.8375];
plot(Hop_count,OursR1,'-.*','LineWidth',1);
hold on

Hop_count   = [5];
OursR2      = [41.06258064516129];
plot(Hop_count,OursR2,'-.p','LineWidth',1);
hold on

Hop_count   = [5];
OursR3      = [37.921149289099525];
plot(Hop_count,OursR3,'-.>','LineWidth',1);
hold on

Hop_count   = [1, 3, 5];
OursR4      = [3.8, 20.206204379562045, 32.166002205199625];
plot(Hop_count,OursR4,'-.<','LineWidth',1);
hold on

grid on

set(gca,'XLim',[0 6]);
set(gca,'XTick',[1 2 3 4 5]);

legend('Set(n)','Set(n/2)','Set(n/4)','Set(n/8)','Set(n/16)','Location','northwest');
xlabel('The number of hops, 1040 nodes');
ylabel('The length of X (bits)');

%Figure for 1440 nodes

h=fig('units','centimeters','width',13,'height',9,'font','Helvetica','fontsize', 12);

Hop_count   = [5];
OursR0      = [53.25];
plot(Hop_count,OursR0,'-.s','LineWidth',1);
hold on

Hop_count   = [5];
OursR1      = [47.682870370370374];
plot(Hop_count,OursR1,'-.*','LineWidth',1);
hold on

Hop_count   = [5];
OursR2      = [42.9215];
plot(Hop_count,OursR2,'-.p','LineWidth',1);
hold on

Hop_count   = [5];
OursR3      = [39.871177863910425];
plot(Hop_count,OursR3,'-.>','LineWidth',1);
hold on

Hop_count   = [1, 3, 5];
OursR4      = [3.875, 21.261061946902654, 34.02772461919122];
plot(Hop_count,OursR4,'-.<','LineWidth',1);
hold on

grid on

set(gca,'XLim',[0 6]);
set(gca,'XTick',[1 2 3 4 5]);

legend('Set(n)','Set(n/2)','Set(n/4)','Set(n/8)','Set(n/16)','Location','northwest');
xlabel('The number of hops, 1440 nodes');
ylabel('The length of X (bits)');
