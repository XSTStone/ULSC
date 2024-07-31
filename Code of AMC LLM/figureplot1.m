clc,clear,close all
load data.mat
for i1=1:length(EbN0indB)
SemanticLoss1(i1)=mean(mean(SemanticLoss(:,:,i1)));
SemanticLossM1(i1)=mean(mean(SemanticLossM(:,:,i1)));
SemanticLossLLM1(i1)=mean(mean(SemanticLossLLM(:,:,i1)));
end
plot(EbN0indB,SemanticLossM1,'b-o')
hold on
plot(EbN0indB,SemanticLoss1,'m-d')
plot(EbN0indB,SemanticLossLLM1,'r-+')
hold on


grid on
title('Two Words in Each Packet')
xlabel('EbN0 (dB)')
ylabel('Semantic Loss')
legend('Communication without Semantic Awareness','SIAC without Semantic Correction','SIAC with Semantic Correction by LLM' ,'location','northeast')



load data1.mat
for i1=1:length(EbN0indB)
SemanticLoss1(i1)=mean(mean(SemanticLoss(:,:,i1)));
SemanticLossM1(i1)=mean(mean(SemanticLossM(:,:,i1)));
SemanticLossLLM1(i1)=mean(mean(SemanticLossLLM(:,:,i1)));
end
figure
plot(EbN0indB,SemanticLossM1,'b-o')
hold on

plot(EbN0indB,SemanticLoss1,'m-d')
plot(EbN0indB,SemanticLossLLM1,'r-+')


grid on
title('A Word in Each Packet')
xlabel('EbN0 (dB)')
ylabel('Semantic Loss')
legend('Communication without Semantic Awareness','SIAC without Semantic Correction','SIAC with Semantic Correction by LLM', 'location','northeast')