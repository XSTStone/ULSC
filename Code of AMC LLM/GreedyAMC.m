function [OptCode,OptModulation,SemanticLossMin,TransmissionDelayMin]=GreedyAMC(wBERT,Nbits,h,ModulationAll,CodingAll,N0,Rs,DelayThreshold,Eb)

OptCode=zeros(1,length(wBERT));
OptModulation=zeros(1,length(wBERT));
Code=CodingAll(end)*ones(1,length(wBERT));
Modulation=ModulationAll(1)*ones(1,length(wBERT));
[SemanticLoss,TransmissionDelay]=ComputeLoss(wBERT,Nbits,Code,Modulation,h,Eb,N0,Rs);
stop=0;
  while TransmissionDelay>DelayThreshold &~stop
         
    for i1=1:length(wBERT)
        for i2=1:2
          ModulationNew=Modulation;
          CodeNew=Code;
          Modulation0=Modulation;
          Code0=Code;
           if i2==1
                Ind=find(ModulationAll==Modulation(i1));
                if Ind<length(ModulationAll)
                     ModulationNew(i1)=ModulationAll(Ind+1);
                end
            else
                Ind=find(CodingAll==Code(i1));
                if Ind>1
                    CodeNew(i1)=CodingAll(Ind-1);
                end
           end
         [SemanticLossTemp(i1,i2),TransmissionDelayTemp(i1,i2)]=ComputeLoss(wBERT,Nbits,CodeNew,ModulationNew,h,Eb,N0,Rs);
        end
    end
    [ind1,ind2]=find(SemanticLossTemp==min(min(SemanticLossTemp)));
    if length(ind1)>1
        [ind1,ind2]=find(TransmissionDelayTemp==min(min(TransmissionDelayTemp)));
    end
     if ind2==1
                Ind=find(ModulationAll==Modulation(ind1(1)));
                if Ind<length(ModulationAll)
                     Modulation(ind1(1))=ModulationAll(Ind+1);
                end
     else
                Ind=find(CodingAll==Code(ind1(1)));
                if Ind>1
                    Code(ind1(1))=CodingAll(Ind-1);
                end
     end
     if length(find(Modulation0~=Modulation))==0&length(find(Code0~=Code))==0
         stop=1;
     end

  end
OptCode=Code;
OptModulation=Modulation;
[SemanticLossMin,TransmissionDelayMin]=ComputeLoss(wBERT,Nbits,Code,Modulation,h,Eb,N0,Rs);
