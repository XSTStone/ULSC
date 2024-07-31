clc,clear,close all
warning off
frame=cell(100,14);
frame_bit=cell(100,14);
%%信源读取与信源编码
for iName=1:100
    filename=['txt_files/',num2str(iName,'%03d'),'.txt'];
    Text=importdata(filename);
    sen=split(Text);
    iK=1;
    for i1=1:length(sen)
        %%分帧打包
        frame{iName,iK}=[frame{iName,iK},sen{i1}];
        if mod(i1,2)==0
            iK=iK+1;
        end
    end
end

for i1=1:100
    for iK=1:14
        if ~isempty(frame{i1,iK})
        abits=de2bi(int8(frame{i1,iK}),8);
        abits=abits.';
        frame_bit{i1,iK}=abits(:).';
        Nbits(i1,iK)=length(frame_bit{i1,iK})+80;%%frame header 80 bits
        end;
    end
end
[wFrame1,wFrame2,wLLM1,wLLM2]=ImportanceMatrixLLM;

% wBERT=wBERT(:,1:5);
% Nbits=Nbits(:,1:5);

ModulationAll=[2,4,16,64,256];%%Modulation
CodingAll=[3,4,5,6];%%Hammard Code
%CodingAll=[2,3,4];%%Reed Muller Code
Eb=1;
Rs=1e6;%%
EbN0indB=0:2:20;
DelayThreshold=1e-3;


for i1=1%:100%0%:100%0
    i1
    MBERT=zeros(size(wFrame2(i1,:)));
    indx=find(wFrame2(i1,:)>0);
    MBERT(indx)=mean(wFrame2(i1,indx))*ones(size(wFrame2(i1,indx)));

    for iChannel=1:10%00%0
            h=1/sqrt(2)*(randn+1j*randn);      
        
            parfor iNoise=1:length(EbN0indB)
                 N0=Eb/(10^(EbN0indB(iNoise)/10));
%                   Code=2*ones(size(MBERT));
%             Modulation=256*ones(size(MBERT));
%             [~,TransmissionDelayLB]=ComputeLoss(MBERT,Nbits(i1,:),Code,Modulation,h,Eb,N0,Rs);
%             Code=6*ones(size(MBERT));
%             Modulation=2*ones(size(MBERT));
%             [~,TransmissionDelayUB]=ComputeLoss(MBERT,Nbits(i1,:),Code,Modulation,h,Eb,N0,Rs);
%             DelayThreshold=(TransmissionDelayUB+TransmissionDelayLB)/2; 
                 %[Code,Modulation,SemanticLoss,TransmissionDelay]=ExhaustiveAMC(wBERT(i1,:),Nbits(i1,:),h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb);
                 [Code,Modulation,SemanticLoss(i1,iChannel,iNoise),TransmissionDelay]=GreedyAMC_New(wFrame2(i1,:),Nbits(i1,:),h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb);
                 [CodeLLM,ModulationLLM,SemanticLossLLM(i1,iChannel,iNoise),TransmissionDelayLLM]=GreedyAMC_New(wLLM2(i1,:),Nbits(i1,:),h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb);
                 [CodeM,ModulationM,SemanticLossM(i1,iChannel,iNoise),TransmissionDelayM]=GreedyAMC_New(MBERT,Nbits(i1,:),h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb);
            end
    end
    save data1.mat
end

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


