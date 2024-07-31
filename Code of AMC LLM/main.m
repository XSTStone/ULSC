clc,clear,close all
warning off
frame=cell(100,20);
frame_bit=cell(100,20);
%%信源读取与信源编码
for iName=1:100
    filename=['data/',num2str(iName),'.txt'];
    Text=importdata(filename);
    sen=split(Text);
    iK=1;
    for i1=1:length(sen)
        %%分帧打包
        frame{iName,iK}=[frame{iName,iK},sen{i1}];
        if mod(i1,5)==0
            iK=iK+1;
        end
    end
end

for i1=1:100
    for iK=1:20
        abits=de2bi(int8(frame{i1,iK}),8);
        abits=abits.';
        frame_bit{i1,iK}=abits(:).';
        Nbits(i1,iK)=length(frame_bit{i1,iK})+80;%%frame header 80 bits
    end
end
wBERT=ImportanceMatrix;

% wBERT=wBERT(:,1:5);
% Nbits=Nbits(:,1:5);

ModulationAll=[2,4,16,64,256];%%Modulation
CodingAll=[3,4,5,6];%%Hammard Code
%CodingAll=[2,3,4];%%Reed Muller Code
Eb=1;
Rs=1e6;%%
EbN0indB=0:2:20;
DelayThreshold=9e-3;

for i1=97:100%0
    i1
    MBERT=mean(wBERT(i1,:))*ones(size(wBERT(i1,:)));
    for iChannel=1:100%0
   iChannel
            h=1/sqrt(2)*(randn+j*randn);       
            parfor iNoise=1:length(EbN0indB)
                 N0=Eb/(10^(EbN0indB(iNoise)/10));
                 %[Code,Modulation,SemanticLoss,TransmissionDelay]=ExhaustiveAMC(wBERT(i1,:),Nbits(i1,:),h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb);
                 [Code,Modulation,SemanticLoss(i1,iChannel,iNoise),TransmissionDelay]=GreedyAMC(wBERT(i1,:),Nbits(i1,:),h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb);
                 [CodeM,ModulationM,SemanticLossM(i1,iChannel,iNoise),TransmissionDelayM]=GreedyAMC(MBERT,Nbits(i1,:),h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb);
            end
    end
    save data.mat
end




