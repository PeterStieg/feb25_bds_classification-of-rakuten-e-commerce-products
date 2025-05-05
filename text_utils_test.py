import pandas as pd
import text_utils
import math

clean_text_df = pd.read_csv('test_utils/clean_text_test.csv', index_col = 0)

def test_clean_description():
    
    # perfect match between both strings
    str1 = 'Ich habe einen String'
    str2 = 'ICH HABE EINEN STRING'
    assert math.isnan(text_utils.clean_description(str1, str2))
    
    # string 1 is longer than string 2
    str1 = 'Ich habe einen String der lang ist'
    str2 = 'ICH HABE EINEN STRING'
    assert math.isnan(text_utils.clean_description(str1, str2))
    
    # string 2 is longer than string 1
    str1 = 'Ich habe einen String'
    str2 = 'ICH HABE EINEN STRING DER LANG IST'
    assert text_utils.clean_description(str1, str2) == 'ICH HABE EINEN STRING DER LANG IST'
    
    # not a good match between both strings
    str1 = 'ich habe keinen match hier'
    str2 = 'ICH HABE EINEN STRING'
    assert text_utils.clean_description(str1, str2) == 'ICH HABE EINEN STRING'

def test_clean_text():
    assert text_utils.text_cleaner(clean_text_df).iloc[0].description_cleaned == 'Pack de 30 pointes de fléchettes électroniques de rechange nylon " Dimple Tip" 2 BA longueur: 26mm couleur: noir Pack de 30 pointes nylon " Dimple Tip" 2 BA longueur: 26mm couleur: noir Vous permet de remplacer les pointes des fléchettes électroniques qui se cassent et s\'abime au fur du temps'
    assert text_utils.text_cleaner(clean_text_df).iloc[1].description_cleaned == 'PAGNA Bewerbungsmappe "Square" DIN A4 blau - für den Markt: Allemand - aus hochwertigem satiniertem Premium-Karton mit 2 Klemmschienen und 2 Klarsichteinschüben mit erhabener Prägung "Bewerbung" auf dem Vorderdeckel 22022-02'
    assert text_utils.text_cleaner(clean_text_df).iloc[2].description_cleaned == 'The Mother hover from Emsco Group is the ultimate Frisbee experience. Each disc is oversized to create a Frisbee that is more stable in the wind. Additionally a patented tie dye process is used to make each disc as unique as its user'
    assert text_utils.text_cleaner(clean_text_df).iloc[3].description_cleaned == ''
    assert text_utils.text_cleaner(clean_text_df).iloc[4].description_cleaned == 'pH moins micro-billes pour spa gonflable 15 kg de la marque Marina.? Permet de réduire le taux de pH de l\'eau.? Optimise l\'efficacité du désinfectant? Améliore votre confort de baignade.? Avec verre doseur pour un dosage facile.? 1/3 de verre doseur pour 1 m³ d\'eau pour diminuer le pH de -02 unités.? Sachet souple refermable de 15 kg.'
    assert text_utils.text_cleaner(clean_text_df).iloc[5].description_cleaned == 'Peinture acrylique. Flacon de 10 ml. Age minimum : 15 ans'
    assert text_utils.text_cleaner(clean_text_df).iloc[6].description_cleaned == "Régate Ma'am Telmar Randonnée extérieure étanche Veste de marche classique CoatPlease consultez le tableau des tailles avant de commander. Si vous n'êtes pas sûr de la taille s'il vous plaît envoyer un message à nous. Description: Fonction: absorption imperméable coupe-vent anti UV agréable à porter l'humidité et perspiration.Color: violet Taille: XLQuality est le premier avec le meilleur service. les clients sont tous nos friends.Fashion design100% Qualité de la marque Newhigh! Style: coupe-vent extérieur et waterprooffStyle: Le sexe féminin Convient pour la saison: printemps automne hiver été. Sport applicables: auto conduite pêche randonnée alpinisme en plein air voyage purpose.Features général: 1.Personal comfortSoft2.Fashion nouvelle Design3.High Qualité 4.Skin convivial et plus de fabrication de comfortablel5.Fine 6.windproof waterproofquick-dryingPackage inclus:"
    assert text_utils.text_cleaner(clean_text_df).iloc[7].description_cleaned == 'Arriver et partir installation passation de pouvoir présider Conseil des ministres sommets internationaux exprimer vœux annuels conférences de presse allocutions innover galette des Rois pèlerinage de Solutré mais aussi assister voyager recevoir commémorer…autant d’actes par lesquels le chef de l’État affirme devant tous sa place et son rôle. À travers un faisceau de représentations fortement emblématiques et au-delà des simples apparences se construisent ainsi son image sa biographie voire sa légende et en miroir celle de la Nation entière.'

    assert text_utils.text_cleaner(clean_text_df).iloc[3].designation_cleaned == 'Dieu Notre Pere Vers Le Jubile De L\'an 2000 Fetes Et Saisons 10 Volumes Numero'

def test_text_pre_processing():
    assert text_utils.text_pre_processing(clean_text_df).iloc[0].merged_text == 'Pack de 30 pointes de fléchettes électroniques de rechange nylon " Dimple Tip" 2 BA longueur: 26mm couleur: noir Pack de 30 pointes nylon " Dimple Tip" 2 BA longueur: 26mm couleur: noir Vous permet de remplacer les pointes des fléchettes électroniques qui se cassent et s\'abime au fur du temps'
    assert text_utils.text_pre_processing(clean_text_df).iloc[1].merged_text == 'PAGNA Bewerbungsmappe "Square" DIN A4 blau - für den Markt: Allemand - aus hochwertigem satiniertem Premium-Karton mit 2 Klemmschienen und 2 Klarsichteinschüben mit erhabener Prägung "Bewerbung" auf dem Vorderdeckel 22022-02'
    assert text_utils.text_pre_processing(clean_text_df).iloc[2].merged_text == 'Emsco Group Mother Hover Super Sized 14.5 Frisbee 14.5 Diameter Mother Hover Super Sized Frisbee - The Mother hover from Emsco Group is the ultimate Frisbee experience. Each disc is oversized to create a Frisbee that is more stable in the wind. Additionally a patented tie dye process is used to make each disc as unique as its user'
    assert text_utils.text_pre_processing(clean_text_df).iloc[3].merged_text == 'Dieu Notre Pere Vers Le Jubile De L\'an 2000 Fetes Et Saisons 10 Volumes Numero'
    assert text_utils.text_pre_processing(clean_text_df).iloc[4].merged_text == 'pH moins micro-billes pour spa gonflable 15 kg de la marque Marina.? Permet de réduire le taux de pH de l\'eau.? Optimise l\'efficacité du désinfectant? Améliore votre confort de baignade.? Avec verre doseur pour un dosage facile.? 1/3 de verre doseur pour 1 m³ d\'eau pour diminuer le pH de -02 unités.? Sachet souple refermable de 15 kg.'
    assert text_utils.text_pre_processing(clean_text_df).iloc[5].merged_text == 'Mini Xf7 - Rouge Mat - Peinture acrylique. Flacon de 10 ml. Age minimum : 15 ans'
    assert text_utils.text_pre_processing(clean_text_df).iloc[6].merged_text == "Régate Ma'am Telmar Randonnée extérieure étanche Veste de marche classique CoatPlease consultez le tableau des tailles avant de commander. Si vous n'êtes pas sûr de la taille s'il vous plaît envoyer un message à nous. Description: Fonction: absorption imperméable coupe-vent anti UV agréable à porter l'humidité et perspiration.Color: violet Taille: XLQuality est le premier avec le meilleur service. les clients sont tous nos friends.Fashion design100% Qualité de la marque Newhigh! Style: coupe-vent extérieur et waterprooffStyle: Le sexe féminin Convient pour la saison: printemps automne hiver été. Sport applicables: auto conduite pêche randonnée alpinisme en plein air voyage purpose.Features général: 1.Personal comfortSoft2.Fashion nouvelle Design3.High Qualité 4.Skin convivial et plus de fabrication de comfortablel5.Fine 6.windproof waterproofquick-dryingPackage inclus:"
    assert text_utils.text_pre_processing(clean_text_df).iloc[7].merged_text == 'Lorsque Le Président Paraît - Arriver et partir installation passation de pouvoir présider Conseil des ministres sommets internationaux exprimer vœux annuels conférences de presse allocutions innover galette des Rois pèlerinage de Solutré mais aussi assister voyager recevoir commémorer…autant d’actes par lesquels le chef de l’État affirme devant tous sa place et son rôle. À travers un faisceau de représentations fortement emblématiques et au-delà des simples apparences se construisent ainsi son image sa biographie voire sa légende et en miroir celle de la Nation entière.'