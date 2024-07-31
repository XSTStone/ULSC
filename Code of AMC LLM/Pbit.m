function Pe=Pbit(Modulation,gamma)
if Modulation==2
    Pe=qfunc(sqrt(2*gamma));
elseif Modulation==4
    Pe=2*qfunc(sqrt(2*gamma))-qfunc(sqrt(gamma))^2;
else
    Pe=4*((sqrt(Modulation)-1)/sqrt(Modulation))*qfunc(sqrt(3*gamma/(Modulation-1)))-4*((sqrt(Modulation)-1)/sqrt(Modulation))^2*qfunc(sqrt(3*gamma/(Modulation-1)))^2;
end
