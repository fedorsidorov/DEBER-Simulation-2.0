// equi.cmd -- to color edges green subject to equiangulation

// Programmer: Ken Brakke, brakke@susqu.edu, http://www.susqu.edu/brakke

// Usage: showequi

showequi := { 
   local ecount, al, nn, cl, dl, el, bl;

   ecount := 0; 
   foreach edge ee do 
   { al := ee.length; 
     nn := 1; 
     foreach ee.facet ff do 
     { foreach ff.edge fe do 
       { if ( fe.id != ee.id ) then 
         { if ( nn == 1 ) then bl := fe.length 
           else if ( nn == 2 ) then  cl := fe.length 
           else if ( nn == 3 ) then  dl := fe.length 
           else if ( nn == 4 ) then  el := fe.length; 
           nn := nn + 1; 
         } 
       }
     }; 
     if ( nn == 5 ) then 
       if (bl^2+cl^2-al^2)/bl/cl + (dl^2+el^2-al^2)/dl/el < -1e-5 then
       { set ee color green;  
         ecount := 1 + ecount 
       } 
   }; 
   printf "Found %g edges.\n",ecount;
   show_expr edges where 1 
}

// End equi.cmd

// Usage: showequi

