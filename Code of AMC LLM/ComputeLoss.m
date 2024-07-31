function [SemanticLoss,TransmissionDelay]=ComputeLoss(wBERT,Nbits,Code,Modulation,h,Eb,N0,Rs)
%  N0=0.01
% % N0=1e-3
% wBERT=wBERT(1,:);
% Nbits=Nbits(1,:);
% Modulation=[4,2,2];
% Code=[6,6,6];
% % 
for l=1:length(wBERT)
  Bl(l)=2^Code(l);
  Rl(l)=Code(l)/(2^Code(l));
  dmin(l)=2^(Code(l)-1);
  Ml(l)=Modulation(l);
end

% %%Reed Muller Code
% for l=1:length(wBERT)
%   m=7;
%   r=Code(l);
%   Bl(l)=2^m;
%   k=0;
% 
%   for i1=0:Code(l)
%       k=k+nchoosek(m,i1);
%   end
%   Rl(l)=k/Bl(l);
%   dmin(l)=2^(m-Code(l));
%   Ml(l)=Modulation(l);
% end

PCblock=zeros(size(wBERT));
for l=1:length(wBERT)
    Cl(l)=floor((dmin(l)-1)/2);
    gamma=abs(h)^2*Eb/N0*Rl(l)*log2(Ml(l));
    for i1=0:Cl(l)
        PCblock(l)=PCblock(l)+nchoosek(Bl(l),i1)*Pbit(Modulation(l),gamma)^i1*(1-Pbit(Modulation(l),gamma))^(Bl(l)-i1);
    end    
    NB(l)=ceil(Nbits(l)/(Rl(l)*Bl(l)));
    PCframe(l)=(PCblock(l)).^NB(l);
end

Pframe=1-PCframe;
for l=1:length(wBERT)
    SL(l)=wBERT(l)*Pframe(l);%/(1-Pframe(l)+eps)*prod(1-Pframe);
    DL(l)=Nbits(l)/(log2(Ml(l))*Rl(l)*Rs);
end
SemanticLoss=sum(SL);
TransmissionDelay=sum(DL);



