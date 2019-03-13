# Siamese_net

Need to refactor the code

Loss Function used : Contrastive Loss Function

	(1-label) * dis^2 + (label) * max(0, margin - dis)^2
label = 0 if same class and label = 1 of different class

observation 1 : consider high margin value to learn the differences between similar looking objects
