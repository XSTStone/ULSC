function [OptCode,OptModulation,SemanticLossMin,TransmissionDelayMin]=ExhaustiveAMC(wBERT,Nbits,h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb)
SemanticLossMin=Inf;
OptCode=zeros(1,length(wBERT));
OptModulation=zeros(1,length(wBERT));
for i1=1:length(ModulationAll)^length(wBERT)
    baseStr1 = dec2base(i1-1,length(ModulationAll),length(wBERT)); 
    I1=str2num(baseStr1(:));
    Modulation=ModulationAll(I1+1);
    for i2=1:length(CodingAll)^length(wBERT)
        baseStr2 = dec2base(i2-1,length(CodingAll),length(wBERT)); 
         I2=str2num(baseStr2(:));
         Code=CodingAll(I2+1);
         
         [SemanticLoss,TransmissionDelay]=ComputeLoss(wBERT,Nbits,Code,Modulation,h,Eb,N0,Rs);
         if TransmissionDelay<DelayThreshold
             if SemanticLoss<SemanticLossMin
                 SemanticLossMin=SemanticLoss;
                 TransmissionDelayMin=TransmissionDelay;
                 OptCode=Code;
                 OptModulation=Modulation;
             end
         end
    end
end
  
   