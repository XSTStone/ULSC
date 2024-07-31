function [OptCode,OptModulation,SemanticLossMin,TransmissionDelayMin]=GreedyAMC_New(wBERT,Nbits,h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb)

OptCode=zeros(1,length(wBERT));
OptModulation=zeros(1,length(wBERT));
% Code=CodingAll(end)*ones(1,length(wBERT));
% Modulation=ModulationAll(1)*ones(1,length(wBERT));
% [SemanticLoss,TransmissionDelay]=ComputeLoss(wBERT,Nbits,Code,Modulation,h,Eb,N0,Rs);
SLmax=Inf;
NRandom=1000;
for i1=1:NRandom
    CodeIndx=randi([1,length(CodingAll)],1,length(wBERT));
    Code=CodingAll(CodeIndx);
    ModIndx=randi([1,length(ModulationAll)],1,length(wBERT));
    Modulation=ModulationAll(ModIndx);
    [SemanticLoss,TransmissionDelay]=ComputeLoss(wBERT,Nbits,Code,Modulation,h,Eb,N0,Rs);
    if TransmissionDelay<DelayThreshold
        if SemanticLoss<SLmax
            SLmax=SemanticLoss;
            OptCode=Code;
            OptModulation=Modulation;
            SemanticLossMin=SemanticLoss;
            TransmissionDelayMin=TransmissionDelay;
        end
    end
end