clc,clear,close all
load data.mat
for i1=1:length(EbN0indB)
SemanticLoss1(i1)=mean(mean(SemanticLoss(:,:,i1)));
SemanticLossM1(i1)=mean(mean(SemanticLossM(:,:,i1)));
SemanticLossLLM1(i1)=mean(mean(SemanticLossLLM(:,:,i1)));
end
semilogy(EbN0indB,SemanticLossLLM1,'r-+')
hold on
semilogy(EbN0indB,SemanticLoss1,'m-d')
semilogy(EbN0indB,SemanticLossM1,'b-o')
xlabel('EbN0 (dB)')
ylabel('Semantic Loss')
legend('Semantic-Aware Communications with Error Correction by LLM','Semantic-Aware Communications without Error Correction by LLM', 'Conventional Communications without Semantic Awareness','location','best')