import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import os
import werkzeug
import logging
from datetime import datetime

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
scaler = None
class_mapping = {}
training_info = {}

# COMPREHENSIVE 40 INDIAN MEDICINAL PLANTS DATABASE
medicinal_plants_database = {
    'aloevera': {
        'common_name': 'Aloe Vera',
        'scientific_name': 'Aloe barbadensis miller',
        'hindi_name': 'घृतकुमारी',
        'family': 'Asphodelaceae',
        'description': 'Aloe Vera is a succulent plant known for its healing gel. Used for thousands of years for skin and digestive health.',
        'medicinal_uses': [
            'Heals burns, wounds and skin irritations',
            'Moisturizes skin and treats sunburn',
            'Aids digestion and relieves constipation',
            'Reduces dental plaque and gum inflammation',
            'Treats acne and skin conditions'
        ],
        'health_benefits': [
            'Soothes and heals skin conditions',
            'Rich in vitamins, minerals and antioxidants',
            'Supports digestive health',
            'Boosts collagen production',
            'Anti-inflammatory properties'
        ],
        'how_to_use': [
            'Apply fresh gel directly on burns and wounds',
            'Drink aloe vera juice for digestive health',
            'Use as face mask for acne and skin care',
            'Apply on hair for dandruff and hair growth'
        ],
        'precautions': [
            'Some people may be allergic to aloe vera',
            'Avoid during pregnancy without consultation',
            'May cause diarrhea in large quantities',
            'Do not consume aloe vera latex in large amounts'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Sweet',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Pitta'
        }
    },

    'amla': {
        'common_name': 'Amla (Indian Gooseberry)',
        'scientific_name': 'Phyllanthus emblica',
        'hindi_name': 'आंवला',
        'family': 'Phyllanthaceae',
        'description': 'Amla is one of the richest natural sources of Vitamin C and a powerful rejuvenator in Ayurveda.',
        'medicinal_uses': [
            'Boosts immunity and prevents colds',
            'Promotes hair growth and prevents graying',
            'Improves eyesight and eye health',
            'Regulates blood sugar levels',
            'Enhances digestion and metabolism'
        ],
        'health_benefits': [
            'Extremely high in Vitamin C and antioxidants',
            'Supports liver function and detoxification',
            'Lowers cholesterol and blood pressure',
            'Anti-aging properties for skin and hair',
            'Improves brain function and memory'
        ],
        'how_to_use': [
            'Eat fresh amla fruit daily for immunity',
            'Use amla powder in hair oils for growth',
            'Drink amla juice for digestive health',
            'Take amla supplements for overall wellness'
        ],
        'precautions': [
            'May cause acidity in some people',
            'Monitor blood sugar if diabetic',
            'Avoid in case of hyperacidity',
            'Consult doctor if taking blood thinners'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sour, Sweet, Bitter, Astringent',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances all three Doshas'
        }
    },

    'amruta_balli': {
        'common_name': 'Giloy (Amruta Balli)',
        'scientific_name': 'Tinospora cordifolia',
        'hindi_name': 'गिलोय',
        'family': 'Menispermaceae',
        'description': 'Giloy is a powerful immunomodulator known as "Amrita" in Ayurveda, meaning root of immortality.',
        'medicinal_uses': [
            'Boosts immunity and fights infections',
            'Reduces fever and manages dengue',
            'Improves digestion and treats acidity',
            'Manages diabetes and blood sugar',
            'Reduces stress and anxiety'
        ],
        'health_benefits': [
            'Powerful antioxidant properties',
            'Anti-inflammatory effects',
            'Liver protective qualities',
            'Anti-arthritic properties',
            'Improves respiratory health'
        ],
        'how_to_use': [
            'Drink giloy juice daily on empty stomach',
            'Take giloy powder with honey or water',
            'Use giloy capsules as supplements',
            'Make giloy kadha for fever and immunity'
        ],
        'precautions': [
            'May lower blood sugar significantly',
            'Avoid during pregnancy and breastfeeding',
            'Consult doctor if taking diabetes medication',
            'May cause constipation in some people'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Astringent',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances all three Doshas'
        }
    },

    'arali': {
        'common_name': 'Oleander (Arali)',
        'scientific_name': 'Nerium oleander',
        'hindi_name': 'कनेर',
        'family': 'Apocynaceae',
        'description': 'Oleander is a beautiful but highly toxic plant used in traditional medicine with extreme caution.',
        'medicinal_uses': [
            'Traditional use for skin diseases',
            'Used in heart conditions under expert supervision',
            'Anti-cancer properties being researched',
            'External application for skin problems'
        ],
        'health_benefits': [
            'Cardiac glycosides for heart conditions',
            'Anti-inflammatory properties',
            'Antibacterial effects',
            'Potential anti-cancer compounds'
        ],
        'how_to_use': [
            'STRICTLY UNDER EXPERT SUPERVISION ONLY',
            'Never consume without proper processing',
            'External applications only with guidance',
            'Traditional formulations by qualified practitioners'
        ],
        'precautions': [
            '⚠️ HIGHLY TOXIC - Can be fatal if ingested',
            '⚠️ Never use without expert guidance',
            '⚠️ Keep away from children and pets',
            '⚠️ Do not self-medicate under any circumstances'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Pungent',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'ashoka': {
        'common_name': 'Ashoka Tree',
        'scientific_name': 'Saraca asoca',
        'hindi_name': 'अशोक',
        'family': 'Fabaceae',
        'description': 'Ashoka tree is known as "sorrow-less tree" and is highly valued in Ayurveda for women health.',
        'medicinal_uses': [
            'Treats menstrual disorders and pain',
            'Helps in uterine health and fibroids',
            'Useful in diarrhea and dysentery',
            'Skin diseases and inflammation',
            'Bleeding disorders management'
        ],
        'health_benefits': [
            'Uterine tonic and regulator',
            'Anti-inflammatory properties',
            'Antioxidant effects',
            'Antimicrobial activity',
            'Analgesic properties'
        ],
        'how_to_use': [
            'Ashoka bark decoction for menstrual issues',
            'Flower juice for diabetes management',
            'Seed powder for kidney stones',
            'Bark extract for skin conditions'
        ],
        'precautions': [
            'Avoid during pregnancy',
            'May cause constipation in high doses',
            'Consult doctor for proper dosage',
            'Monitor blood sugar levels'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Astringent',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'ashwagandha': {
        'common_name': 'Ashwagandha (Winter Cherry)',
        'scientific_name': 'Withania somnifera',
        'hindi_name': 'अश्वगंधा',
        'family': 'Solanaceae',
        'description': 'Ashwagandha is a powerful adaptogen known for its stress-relieving and rejuvenating properties.',
        'medicinal_uses': [
            'Reduces stress and anxiety',
            'Improves energy and stamina',
            'Enhances brain function and memory',
            'Boosts male reproductive health',
            'Supports immune system'
        ],
        'health_benefits': [
            'Adaptogenic properties help manage stress',
            'Increases muscle strength and endurance',
            'Improves sleep quality',
            'Anti-inflammatory and antioxidant effects',
            'Neuroprotective benefits'
        ],
        'how_to_use': [
            'Take ashwagandha powder with milk at night',
            'Use capsules or tablets as supplements',
            'Add to smoothies or warm beverages',
            'Consult Ayurvedic practitioner for dosage'
        ],
        'precautions': [
            'Avoid during pregnancy',
            'May interact with sedatives',
            'Consult doctor if taking thyroid medication',
            'May cause stomach upset in some people'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Astringent',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Vata and Kapha'
        }
    },

    'avacado': {
        'common_name': 'Avocado',
        'scientific_name': 'Persea americana',
        'hindi_name': 'एवोकाडो',
        'family': 'Lauraceae',
        'description': 'Avocado is a nutrient-dense fruit rich in healthy fats, vitamins, and minerals.',
        'medicinal_uses': [
            'Supports heart health and cholesterol',
            'Promotes skin and hair health',
            'Aids weight management',
            'Improves digestion',
            'Rich source of antioxidants'
        ],
        'health_benefits': [
            'High in healthy monounsaturated fats',
            'Rich in fiber for digestive health',
            'Contains vitamins E, C, K, and B6',
            'Potassium-rich for blood pressure',
            'Anti-inflammatory properties'
        ],
        'how_to_use': [
            'Eat fresh avocado as fruit',
            'Use in salads and sandwiches',
            'Make avocado smoothies',
            'Apply avocado paste on skin and hair'
        ],
        'precautions': [
            'High in calories - moderate consumption',
            'May cause allergies in some people',
            'Avoid if allergic to latex',
            'Monitor portion size for weight management'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Vata and Pitta'
        }
    },

    'bamboo': {
        'common_name': 'Bamboo',
        'scientific_name': 'Bambusoideae',
        'hindi_name': 'बांस',
        'family': 'Poaceae',
        'description': 'Bamboo has various medicinal uses, especially bamboo shoots and leaves in traditional medicine.',
        'medicinal_uses': [
            'Respiratory disorders treatment',
            'Wound healing and skin conditions',
            'Arthritis and joint pain relief',
            'Digestive health improvement',
            'Fever and infection management'
        ],
        'health_benefits': [
            'Rich in silica for bone health',
            'Antioxidant properties',
            'Anti-inflammatory effects',
            'High in dietary fiber',
            'Low calorie nutrient source'
        ],
        'how_to_use': [
            'Bamboo shoot curry for digestion',
            'Bamboo leaf tea for respiratory issues',
            'Bamboo sap for skin conditions',
            'Bamboo salt for cooking'
        ],
        'precautions': [
            'Proper cooking required for shoots',
            'Some species may contain toxins',
            'Consult expert for medicinal use',
            'Avoid raw bamboo consumption'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet, Astringent',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'basale': {
        'common_name': 'Malabar Spinach (Basale)',
        'scientific_name': 'Basella alba',
        'hindi_name': 'पोई साग',
        'family': 'Basellaceae',
        'description': 'Basale is a nutritious leafy vegetable with cooling properties and medicinal benefits.',
        'medicinal_uses': [
            'Treats constipation and digestive issues',
            'Cooling effect on body',
            'Rich in iron for anemia',
            'Promotes wound healing',
            'Anti-inflammatory properties'
        ],
        'health_benefits': [
            'High in vitamins A, C, and iron',
            'Rich in fiber for digestion',
            'Mucilaginous properties soothe digestion',
            'Low in calories for weight management',
            'Antioxidant properties'
        ],
        'how_to_use': [
            'Cook as vegetable curry',
            'Make basale juice for constipation',
            'Use in soups and stews',
            'Apply leaf paste on wounds'
        ],
        'precautions': [
            'Generally safe when cooked',
            'May cause oxalate issues in sensitive people',
            'Cook properly to reduce oxalates',
            'Moderate consumption recommended'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet, Astringent',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Pitta and Vata'
        }
    },

    'betel': {
        'common_name': 'Betel Leaf (Paan)',
        'scientific_name': 'Piper betle',
        'hindi_name': 'पान',
        'family': 'Piperaceae',
        'description': 'Betel leaf has digestive and medicinal properties, traditionally used in Ayurveda.',
        'medicinal_uses': [
            'Improves digestion and appetite',
            'Respiratory problems relief',
            'Wound healing and antiseptic',
            'Oral health and fresh breath',
            'Headache and pain relief'
        ],
        'health_benefits': [
            'Antimicrobial properties',
            'Antioxidant effects',
            'Anti-inflammatory benefits',
            'Digestive stimulant',
            'Respiratory health support'
        ],
        'how_to_use': [
            'Chew fresh leaf after meals for digestion',
            'Apply leaf paste on wounds',
            'Use in aromatherapy for headaches',
            'Betel leaf juice for cough'
        ],
        'precautions': [
            'Avoid with tobacco and areca nut',
            'May cause allergies in some',
            'Moderate use recommended',
            'Consult doctor for medicinal use'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Pungent, Bitter',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'betel_nut': {
        'common_name': 'Betel Nut (Areca Nut)',
        'scientific_name': 'Areca catechu',
        'hindi_name': 'सुपारी',
        'family': 'Arecaceae',
        'description': 'Betel nut has traditional medicinal uses but is known for its stimulant properties and health risks.',
        'medicinal_uses': [
            'Traditional digestive aid',
            'Mild stimulant properties',
            'Astringent for oral health',
            'Traditional worm treatment',
            'Skin conditions in small amounts'
        ],
        'health_benefits': [
            'Mild stimulant effect',
            'Astringent properties',
            'Traditional digestive aid',
            'Antimicrobial effects in small doses'
        ],
        'how_to_use': [
            'STRICTLY LIMITED MEDICINAL USE ONLY',
            'Traditional formulations under guidance',
            'Small amounts for digestive issues',
            'External applications only'
        ],
        'precautions': [
            '⚠️ Known carcinogen with long-term use',
            '⚠️ Highly addictive substance',
            '⚠️ Increases risk of oral cancer',
            '⚠️ Avoid regular consumption',
            '⚠️ Not recommended for medicinal use'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Pungent, Bitter',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Increases Pitta, balances Kapha'
        }
    },

    'brahmi': {
        'common_name': 'Brahmi (Water Hyssop)',
        'scientific_name': 'Bacopa monnieri',
        'hindi_name': 'ब्राह्मी',
        'family': 'Plantaginaceae',
        'description': 'Brahmi is renowned for enhancing brain function, memory, and cognitive abilities.',
        'medicinal_uses': [
            'Improves memory and learning ability',
            'Reduces anxiety and stress',
            'Enhances concentration and focus',
            'Treats epilepsy and seizures',
            'Promotes hair growth'
        ],
        'health_benefits': [
            'Neuroprotective properties for brain health',
            'Antioxidant effects protect brain cells',
            'Improves blood circulation to brain',
            'Calms nervous system',
            'Anti-inflammatory properties'
        ],
        'how_to_use': [
            'Take brahmi powder with ghee or honey',
            'Use brahmi oil for head massage',
            'Drink brahmi tea for mental clarity',
            'Apply brahmi paste on hair for growth'
        ],
        'precautions': [
            'May cause digestive issues in high doses',
            'Consult doctor if taking thyroid medication',
            'Avoid during pregnancy without consultation',
            'May interact with sedative medications'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Sweet',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances all three Doshas'
        }
    },

    'castor': {
        'common_name': 'Castor Plant',
        'scientific_name': 'Ricinus communis',
        'hindi_name': 'अरंडी',
        'family': 'Euphorbiaceae',
        'description': 'Castor plant is known for its medicinal oil but all parts of plant contain toxic compounds.',
        'medicinal_uses': [
            'Castor oil for constipation relief',
            'Anti-inflammatory for arthritis',
            'Skin conditions treatment',
            'Hair growth promotion',
            'Labor induction in traditional medicine'
        ],
        'health_benefits': [
            'Powerful laxative properties',
            'Anti-inflammatory effects',
            'Antimicrobial properties',
            'Moisturizing for skin and hair',
            'Pain relief for joints'
        ],
        'how_to_use': [
            'Castor oil for constipation (small doses)',
            'External application for arthritis',
            'Hair oil for growth and strength',
            'Skin moisturizer and healer'
        ],
        'precautions': [
            '⚠️ Seeds are HIGHLY TOXIC - never consume',
            'Castor oil only in recommended doses',
            'Avoid during pregnancy',
            'May cause allergic reactions',
            'Consult doctor before internal use'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet, Pungent, Bitter',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Vata'
        }
    },

    'curry_leaf': {
        'common_name': 'Curry Leaf',
        'scientific_name': 'Murraya koenigii',
        'hindi_name': 'कढ़ी पत्ता',
        'family': 'Rutaceae',
        'description': 'Curry leaves are aromatic herbs used in cooking with significant medicinal properties.',
        'medicinal_uses': [
            'Aids digestion and relieves nausea',
            'Promotes hair growth and prevents graying',
            'Lowers blood sugar levels',
            'Improves eyesight',
            'Reduces cholesterol'
        ],
        'health_benefits': [
            'Rich in iron and folic acid',
            'Contains antioxidants that fight free radicals',
            'Anti-diabetic properties',
            'Supports weight loss',
            'Liver protective effects'
        ],
        'how_to_use': [
            'Add fresh leaves to curries and dishes',
            'Make curry leaf chutney for digestion',
            'Use curry leaf oil for hair massage',
            'Drink curry leaf tea for diabetes'
        ],
        'precautions': [
            'Generally safe when used in cooking',
            'Medicinal doses should be monitored',
            'May lower blood sugar significantly',
            'Consult doctor if taking diabetes medication'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Pungent, Bitter',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'doddapatre': {
        'common_name': 'Indian Borage (Doddapatre)',
        'scientific_name': 'Coleus amboinicus',
        'hindi_name': 'पथर्चुर',
        'family': 'Lamiaceae',
        'description': 'Doddapatre is an aromatic herb with strong medicinal properties for respiratory and digestive health.',
        'medicinal_uses': [
            'Treats cough and cold effectively',
            'Relieves asthma and bronchitis',
            'Aids digestion and reduces flatulence',
            'Kidney stone treatment',
            'Skin conditions and wounds'
        ],
        'health_benefits': [
            'Expectorant properties for respiratory issues',
            'Antimicrobial and antibacterial',
            'Anti-inflammatory effects',
            'Rich in vitamins and minerals',
            'Diuretic properties'
        ],
        'how_to_use': [
            'Leaf juice with honey for cough',
            'Chew leaves for digestive issues',
            'Apply leaf paste on wounds',
            'Make tea for respiratory problems'
        ],
        'precautions': [
            'Avoid during pregnancy',
            'May cause skin irritation in some',
            'Moderate use recommended',
            'Consult doctor for kidney problems'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Pungent, Bitter',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'ekka': {
        'common_name': 'Crown Flower (Ekka)',
        'scientific_name': 'Calotropis gigantea',
        'hindi_name': 'आक',
        'family': 'Apocynaceae',
        'description': 'Ekka is a medicinal plant with toxic properties, used cautiously in traditional medicine.',
        'medicinal_uses': [
            'Skin diseases treatment',
            'Digestive disorders in small doses',
            'Traditional use for asthma',
            'Wound healing properties',
            'Anti-inflammatory effects'
        ],
        'health_benefits': [
            'Antimicrobial properties',
            'Anti-inflammatory effects',
            'Analgesic properties',
            'Antioxidant activity',
            'Traditional pain relief'
        ],
        'how_to_use': [
            'STRICTLY UNDER EXPERT GUIDANCE',
            'External applications for skin',
            'Traditional formulations only',
            'Never consume raw plant parts'
        ],
        'precautions': [
            '⚠️ MILKY LATEX IS TOXIC',
            '⚠️ Never consume without processing',
            '⚠️ Keep away from eyes and mouth',
            '⚠️ Use only under qualified supervision'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Pungent',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'ganike': {
        'common_name': 'Black Nightshade (Ganike)',
        'scientific_name': 'Solanum nigrum',
        'hindi_name': 'मकोय',
        'family': 'Solanaceae',
        'description': 'Ganike is a medicinal plant used in traditional medicine with both nutritional and therapeutic values.',
        'medicinal_uses': [
            'Fever and inflammation reduction',
            'Liver disorders treatment',
            'Skin diseases and wounds',
            'Digestive issues management',
            'Respiratory problems relief'
        ],
        'health_benefits': [
            'Antipyretic properties',
            'Anti-inflammatory effects',
            'Hepatoprotective qualities',
            'Antioxidant properties',
            'Diuretic effects'
        ],
        'how_to_use': [
            'Cooked leaves as vegetable',
            'Leaf juice for fever',
            'Paste application for skin',
            'Decoction for liver health'
        ],
        'precautions': [
            'Unripe berries may be toxic',
            'Proper identification required',
            'Cook thoroughly before consumption',
            'Consult expert for medicinal use'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Pungent',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'gauva': {
        'common_name': 'Guava',
        'scientific_name': 'Psidium guajava',
        'hindi_name': 'अमरूद',
        'family': 'Myrtaceae',
        'description': 'Guava is a tropical fruit rich in vitamins and antioxidants with numerous health benefits.',
        'medicinal_uses': [
            'Treats diarrhea and dysentery',
            'Manages blood sugar levels',
            'Improves heart health',
            'Boosts immunity',
            'Aids weight loss'
        ],
        'health_benefits': [
            'Rich in Vitamin C and antioxidants',
            'High fiber content for digestion',
            'Low glycemic index for diabetics',
            'Potassium for blood pressure',
            'Anti-inflammatory properties'
        ],
        'how_to_use': [
            'Eat fresh fruit for vitamins',
            'Guava leaf tea for diarrhea',
            'Leaf extract for diabetes',
            'Fruit in salads and juices'
        ],
        'precautions': [
            'May cause bloating if eaten in excess',
            'Monitor blood sugar if diabetic',
            'Eat ripe fruit for best benefits',
            'Wash thoroughly before eating'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet, Astringent',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'geranium': {
        'common_name': 'Geranium',
        'scientific_name': 'Pelargonium graveolens',
        'hindi_name': 'गेरेनियम',
        'family': 'Geraniaceae',
        'description': 'Geranium is known for its aromatic leaves and essential oil with therapeutic properties.',
        'medicinal_uses': [
            'Skin conditions and acne treatment',
            'Stress and anxiety relief',
            'Anti-inflammatory for wounds',
            'Hormonal balance support',
            'Respiratory issues relief'
        ],
        'health_benefits': [
            'Antimicrobial properties',
            'Anti-inflammatory effects',
            'Astringent qualities',
            'Antidepressant properties',
            'Antiseptic for wounds'
        ],
        'how_to_use': [
            'Geranium essential oil for aromatherapy',
            'Leaf paste for skin conditions',
            'Tea for stress relief',
            'Steam inhalation for respiratory issues'
        ],
        'precautions': [
            'Essential oil should be diluted',
            'May cause skin irritation in some',
            'Avoid during pregnancy',
            'Patch test before skin application'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Astringent, Bitter',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'henna': {
        'common_name': 'Henna (Mehndi)',
        'scientific_name': 'Lawsonia inermis',
        'hindi_name': 'मेहंदी',
        'family': 'Lythraceae',
        'description': 'Henna is famous for its natural dyeing properties and cooling medicinal effects.',
        'medicinal_uses': [
            'Natural hair dye and conditioner',
            'Treats skin diseases and infections',
            'Cooling effect for headaches and burns',
            'Anti-fungal properties for feet',
            'Soothes inflammatory conditions'
        ],
        'health_benefits': [
            'Natural cooling agent for body',
            'Antibacterial and antifungal properties',
            'Conditions hair and prevents dandruff',
            'Heals wounds and burns',
            'Anti-inflammatory effects'
        ],
        'how_to_use': [
            'Apply henna paste on hair for coloring',
            'Use henna paste on burns for relief',
            'Apply on feet for fungal infections',
            'Use as natural hand and body art'
        ],
        'precautions': [
            'Test for allergy before use',
            'Avoid chemical mixed henna',
            'May dry hair if used frequently',
            'Use natural henna without additives'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Astringent, Bitter',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'hibiscus': {
        'common_name': 'Hibiscus (Gudhal)',
        'scientific_name': 'Hibiscus rosa-sinensis',
        'hindi_name': 'गुड़हल',
        'family': 'Malvaceae',
        'description': 'Hibiscus is known for its beautiful flowers and significant medicinal properties for hair and heart health.',
        'medicinal_uses': [
            'Promotes hair growth and prevents graying',
            'Lowers blood pressure',
            'Supports liver health',
            'Relieves menstrual cramps',
            'Improves skin health'
        ],
        'health_benefits': [
            'Rich in antioxidants and Vitamin C',
            'Natural diuretic properties',
            'Lowers cholesterol levels',
            'Anti-inflammatory effects',
            'Hair conditioning properties'
        ],
        'how_to_use': [
            'Use hibiscus powder in hair packs',
            'Drink hibiscus tea for blood pressure',
            'Apply hibiscus paste on hair for growth',
            'Use flower extract in skin care'
        ],
        'precautions': [
            'Avoid during pregnancy',
            'May lower blood pressure significantly',
            'Consult doctor if taking blood pressure medication',
            'May interact with diabetes medications'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sour, Astringent',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sour',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'hongue': {
        'common_name': 'Indian Beech (Hongue)',
        'scientific_name': 'Pongamia pinnata',
        'hindi_name': 'करंज',
        'family': 'Fabaceae',
        'description': 'Hongue is a traditional medicinal plant with various therapeutic applications.',
        'medicinal_uses': [
            'Skin diseases treatment',
            'Rheumatism and joint pain',
            'Digestive disorders',
            'Respiratory problems',
            'Wound healing'
        ],
        'health_benefits': [
            'Anti-inflammatory properties',
            'Antimicrobial effects',
            'Analgesic properties',
            'Antioxidant activity',
            'Wound healing properties'
        ],
        'how_to_use': [
            'Oil application for skin conditions',
            'Leaf paste for wounds',
            'Seed powder for digestive issues',
            'Bark decoction for rheumatism'
        ],
        'precautions': [
            'Seeds are toxic if consumed raw',
            'Use only under guidance',
            'May cause skin irritation',
            'Avoid internal use without processing'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Pungent',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'insulin': {
        'common_name': 'Insulin Plant',
        'scientific_name': 'Costus igneus',
        'hindi_name': 'इन्सुलिन प्लांट',
        'family': 'Costaceae',
        'description': 'Insulin plant is known for its anti-diabetic properties and blood sugar regulating effects.',
        'medicinal_uses': [
            'Diabetes management',
            'Blood sugar regulation',
            'Antioxidant properties',
            'Urinary tract health',
            'Liver protection'
        ],
        'health_benefits': [
            'Lowers blood glucose levels',
            'Rich in antioxidants',
            'Diuretic properties',
            'Anti-inflammatory effects',
            'Hepatoprotective qualities'
        ],
        'how_to_use': [
            'Chew fresh leaves daily',
            'Make leaf tea for diabetes',
            'Leaf powder with water',
            'Consult doctor for dosage'
        ],
        'precautions': [
            'Monitor blood sugar regularly',
            'Consult doctor before use',
            'May interact with diabetes medication',
            'Start with small doses'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Pitta'
        }
    },

    'jasmine': {
        'common_name': 'Jasmine',
        'scientific_name': 'Jasminum officinale',
        'hindi_name': 'चमेली',
        'family': 'Oleaceae',
        'description': 'Jasmine is renowned for its fragrant flowers and therapeutic properties in aromatherapy.',
        'medicinal_uses': [
            'Stress and anxiety relief',
            'Skin care and complexion',
            'Headache and pain relief',
            'Antiseptic for wounds',
            'Mood enhancement'
        ],
        'health_benefits': [
            'Antidepressant properties',
            'Antiseptic and antimicrobial',
            'Anti-inflammatory effects',
            'Relaxing and calming',
            'Aphrodisiac properties'
        ],
        'how_to_use': [
            'Jasmine tea for relaxation',
            'Essential oil for aromatherapy',
            'Flower paste for skin care',
            'Jasmine water as toner'
        ],
        'precautions': [
            'Generally safe in moderation',
            'May cause allergies in some',
            'Essential oil should be diluted',
            'Avoid during pregnancy in large amounts'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Astringent',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'lemon': {
        'common_name': 'Lemon (Nimbu)',
        'scientific_name': 'Citrus limon',
        'hindi_name': 'नींबू',
        'family': 'Rutaceae',
        'description': 'Lemon is a citrus fruit rich in Vitamin C with powerful detoxifying properties.',
        'medicinal_uses': [
            'Boosts immunity and prevents scurvy',
            'Aids digestion and weight loss',
            'Purifies blood and detoxifies body',
            'Improves skin health and complexion',
            'Prevents kidney stones'
        ],
        'health_benefits': [
            'High in Vitamin C and antioxidants',
            'Alkalizing effect on body',
            'Supports liver detoxification',
            'Antibacterial and antiviral properties',
            'Rich in potassium'
        ],
        'how_to_use': [
            'Drink warm lemon water every morning',
            'Use lemon juice in salads and dishes',
            'Apply lemon juice on skin for glow',
            'Use lemon with honey for sore throat'
        ],
        'precautions': [
            'May erode tooth enamel - rinse mouth after use',
            'Can cause heartburn in some people',
            'Dilute properly before consumption',
            'Avoid on open wounds'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sour',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Sour',
            'Dosha': 'Increases Pitta, balances Kapha and Vata'
        }
    },

    'lemon_grass': {
        'common_name': 'Lemon Grass',
        'scientific_name': 'Cymbopogon citratus',
        'hindi_name': 'लेमन ग्रास',
        'family': 'Poaceae',
        'description': 'Lemon grass is an aromatic herb with refreshing citrus flavor and medicinal properties.',
        'medicinal_uses': [
            'Digestive issues and bloating',
            'Fever and infection reduction',
            'Stress and anxiety relief',
            'Cholesterol management',
            'Detoxification and cleansing'
        ],
        'health_benefits': [
            'Antimicrobial and antibacterial',
            'Anti-inflammatory properties',
            'Rich in antioxidants',
            'Diuretic effects',
            'Analgesic properties'
        ],
        'how_to_use': [
            'Lemon grass tea for digestion',
            'Essential oil for aromatherapy',
            'Fresh stalks in cooking',
            'Poultice for pain relief'
        ],
        'precautions': [
            'Generally safe in food amounts',
            'May cause allergies in some',
            'Avoid during pregnancy in large amounts',
            'Consult doctor for medicinal use'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Pungent, Bitter',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'mango': {
        'common_name': 'Mango (Aam)',
        'scientific_name': 'Mangifera indica',
        'hindi_name': 'आम',
        'family': 'Anacardiaceae',
        'description': 'Mango is the king of fruits with numerous health benefits beyond its delicious taste.',
        'medicinal_uses': [
            'Boosts immunity with high Vitamin C',
            'Promotes eye health with Vitamin A',
            'Aids digestion and prevents constipation',
            'Lowers cholesterol',
            'Alkalizes whole body'
        ],
        'health_benefits': [
            'Rich in vitamins A, C, and E',
            'High in fiber for digestive health',
            'Contains antioxidants like quercetin',
            'Supports heart health',
            'Anti-cancer properties'
        ],
        'how_to_use': [
            'Eat ripe mangoes in season',
            'Use raw mango in chutneys and pickles',
            'Drink mango juice for energy',
            'Apply mango pulp on skin for glow'
        ],
        'precautions': [
            'High in natural sugars - moderate if diabetic',
            'Some people may be allergic to mango skin',
            'Avoid unripe mangoes in large quantities',
            'Wash thoroughly before eating'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet, Sour (unripe)',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Increases Pitta in excess, balances Vata'
        }
    },

    'mint': {
        'common_name': 'Mint (Pudina)',
        'scientific_name': 'Mentha spicata',
        'hindi_name': 'पुदीना',
        'family': 'Lamiaceae',
        'description': 'Mint is a refreshing herb with cooling properties and numerous health benefits.',
        'medicinal_uses': [
            'Relieves indigestion and IBS symptoms',
            'Clears respiratory congestion',
            'Soothes headaches and migraines',
            'Freshens breath naturally',
            'Relieves muscle pain'
        ],
        'health_benefits': [
            'Antioxidant and anti-inflammatory properties',
            'Antibacterial effects for oral health',
            'Calms stomach muscles and relieves gas',
            'Cooling effect on body',
            'Analgesic properties'
        ],
        'how_to_use': [
            'Chew fresh leaves for fresh breath',
            'Make mint tea for digestion',
            'Apply mint paste for headache relief',
            'Use in salads and chutneys'
        ],
        'precautions': [
            'Generally safe in food amounts',
            'Large amounts may cause heartburn',
            'Avoid in infants and young children',
            'May interact with certain medications'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Pungent',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'nagadali': {
        'common_name': 'Indian Snakeroot (Nagadali)',
        'scientific_name': 'Rauvolfia serpentina',
        'hindi_name': 'सर्पगंधा',
        'family': 'Apocynaceae',
        'description': 'Nagadali is a traditional medicinal plant known for its sedative and antihypertensive properties.',
        'medicinal_uses': [
            'High blood pressure management',
            'Mental disorders and insomnia',
            'Snake bite treatment (traditional)',
            'Anxiety and stress relief',
            'Fever and digestive issues'
        ],
        'health_benefits': [
            'Antihypertensive properties',
            'Sedative and tranquilizing effects',
            'Antipsychotic properties',
            'Antipyretic effects',
            'Traditional use for various ailments'
        ],
        'how_to_use': [
            'STRICTLY UNDER MEDICAL SUPERVISION',
            'Ayurvedic formulations only',
            'Never self-medicate',
            'Traditional preparations by experts'
        ],
        'precautions': [
            '⚠️ POTENT MEDICINE - Requires expert guidance',
            '⚠️ May cause serious side effects',
            '⚠️ Not for self-medication',
            '⚠️ Monitor blood pressure regularly'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Astringent',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'neem': {
        'common_name': 'Neem',
        'scientific_name': 'Azadirachta indica',
        'hindi_name': 'नीम',
        'family': 'Meliaceae',
        'description': 'Neem is called the "Village Pharmacy" due to its numerous medicinal properties. Excellent for skin and blood purification.',
        'medicinal_uses': [
            'Powerful blood purifier and detoxifier',
            'Treats skin diseases like eczema and psoriasis',
            'Natural pesticide and insect repellent',
            'Helps in diabetes management',
            'Dental care and oral hygiene'
        ],
        'health_benefits': [
            'Purifies blood and removes toxins',
            'Heals skin conditions and wounds',
            'Boosts immune system function',
            'Supports liver health and digestion',
            'Antibacterial and antifungal properties'
        ],
        'how_to_use': [
            'Chew neem leaves for blood purification',
            'Apply neem paste on skin for acne and infections',
            'Use neem oil for hair growth and dandruff',
            'Drink neem tea for diabetes control'
        ],
        'precautions': [
            'Avoid during pregnancy and breastfeeding',
            'May cause infertility in high doses',
            'Can interact with diabetes medications',
            'May lower blood sugar significantly'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Pitta'
        }
    },

    'nithyapushpa': {
        'common_name': 'Periwinkle (Nithyapushpa)',
        'scientific_name': 'Catharanthus roseus',
        'hindi_name': 'सदाबहार',
        'family': 'Apocynaceae',
        'description': 'Periwinkle is known for its beautiful flowers and important medicinal compounds used in cancer treatment.',
        'medicinal_uses': [
            'Diabetes management',
            'Traditional use for cancer',
            'Blood pressure regulation',
            'Antimicrobial properties',
            'Memory enhancement'
        ],
        'health_benefits': [
            'Anti-diabetic properties',
            'Source of anti-cancer compounds',
            'Antihypertensive effects',
            'Antioxidant properties',
            'Cognitive enhancement'
        ],
        'how_to_use': [
            'Leaf extract for diabetes',
            'Traditional formulations only',
            'Consult doctor for proper use',
            'Never self-medicate for serious conditions'
        ],
        'precautions': [
            '⚠️ Contains potent alkaloids',
            '⚠️ Use only under medical supervision',
            '⚠️ May interact with medications',
            '⚠️ Not for self-treatment of cancer'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Pitta'
        }
    },
    
    'nooni': {
        'common_name': 'Noni Fruit',
        'scientific_name': 'Morinda citrifolia',
        'hindi_name': 'नोनी',
        'family': 'Rubiaceae',
        'description': 'Noni fruit is known for its immune-boosting properties and has been used in traditional Polynesian medicine for centuries.',
        'medicinal_uses': [
            'Boosts immune system function',
            'Reduces inflammation and pain',
            'Improves skin health and conditions',
            'Supports cardiovascular health',
            'Aids digestion and gut health'
        ],
        'health_benefits': [
            'Rich in antioxidants and phytochemicals',
            'Anti-inflammatory properties',
            'Antimicrobial and antibacterial effects',
            'Analgesic (pain-relieving) properties',
            'Immune-modulating effects'
        ],
        'how_to_use': [
            'Drink noni juice on empty stomach',
            'Apply noni pulp on skin for conditions',
            'Use noni capsules as supplements',
            'Noni leaf tea for internal health'
        ],
        'precautions': [
            'May interact with blood pressure medications',
            'Can affect liver enzymes - monitor with liver conditions',
            'High potassium content - caution with kidney problems',
            'Start with small doses to check tolerance'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Pungent',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'pappaya': {
        'common_name': 'Papaya',
        'scientific_name': 'Carica papaya',
        'hindi_name': 'पपीता',
        'family': 'Caricaceae',
        'description': 'Papaya is a tropical fruit rich in digestive enzymes and antioxidants, known for its numerous health benefits.',
        'medicinal_uses': [
            'Improves digestion and relieves constipation',
            'Treats skin wounds and burns',
            'Boosts immunity with Vitamin C',
            'Supports heart health',
            'Anti-parasitic properties'
        ],
        'health_benefits': [
            'Contains papain enzyme for protein digestion',
            'Rich in Vitamin C and antioxidants',
            'High fiber content for digestive health',
            'Anti-inflammatory properties',
            'Wound healing capabilities'
        ],
        'how_to_use': [
            'Eat ripe papaya for digestion',
            'Apply raw papaya on wounds and burns',
            'Papaya seed juice for parasites',
            'Papaya leaf tea for dengue fever'
        ],
        'precautions': [
            'Unripe papaya may cause uterine contractions - avoid during pregnancy',
            'Papaya seeds in large amounts may be toxic',
            'May cause allergies in latex-sensitive individuals',
            'Moderate consumption recommended'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Vata and Kapha'
        }
    },

    'pepper': {
        'common_name': 'Black Pepper',
        'scientific_name': 'Piper nigrum',
        'hindi_name': 'काली मिर्च',
        'family': 'Piperaceae',
        'description': 'Black pepper is not just a spice but also a powerful medicinal herb with numerous therapeutic properties.',
        'medicinal_uses': [
            'Improves digestion and nutrient absorption',
            'Relieves cold and respiratory issues',
            'Anti-inflammatory and pain relief',
            'Antioxidant properties',
            'Enhances bioavailability of other herbs'
        ],
        'health_benefits': [
            'Contains piperine for enhanced nutrient absorption',
            'Rich in antioxidants',
            'Anti-inflammatory properties',
            'Antimicrobial effects',
            'Improves cognitive function'
        ],
        'how_to_use': [
            'Add to food for digestion enhancement',
            'Pepper tea with honey for cold',
            'Pepper powder with ghee for joint pain',
            'Inhalation for sinus congestion'
        ],
        'precautions': [
            'May cause gastrointestinal irritation in large amounts',
            'Avoid in cases of gastric ulcers',
            'May interact with certain medications',
            'Use in moderation'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Pungent',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'pomegranate': {
        'common_name': 'Pomegranate (Anar)',
        'scientific_name': 'Punica granatum',
        'hindi_name': 'अनार',
        'family': 'Lythraceae',
        'description': 'Pomegranate is a superfruit packed with antioxidants and numerous health benefits for heart and overall health.',
        'medicinal_uses': [
            'Improves heart health and circulation',
            'Lowers blood pressure',
            'Fights cancer cells',
            'Improves digestion',
            'Boosts immunity'
        ],
        'health_benefits': [
            'Extremely high in antioxidants',
            'Anti-inflammatory properties',
            'Rich in Vitamin C and K',
            'Supports joint health',
            'Improves memory and brain function'
        ],
        'how_to_use': [
            'Eat fresh pomegranate seeds daily',
            'Drink pomegranate juice for heart health',
            'Use in salads and desserts',
            'Apply pomegranate paste on skin'
        ],
        'precautions': [
            'May interact with blood pressure medications',
            'High in natural sugars - moderate if diabetic',
            'Some people may be allergic',
            'May affect certain cholesterol medications'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet, Sour, Astringent',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'raktachandini': {
        'common_name': 'Red Sandalwood (Raktachandini)',
        'scientific_name': 'Pterocarpus santalinus',
        'hindi_name': 'रक्तचंदन',
        'family': 'Fabaceae',
        'description': 'Red Sandalwood is prized for its medicinal properties and is used in traditional medicine for skin and inflammatory conditions.',
        'medicinal_uses': [
            'Skin diseases and inflammation',
            'Fever and headache relief',
            'Digestive disorders',
            'Bleeding disorders',
            'Diabetes management'
        ],
        'health_benefits': [
            'Anti-inflammatory properties',
            'Antipyretic (fever-reducing) effects',
            'Astringent and cooling properties',
            'Antioxidant activity',
            'Blood purifying properties'
        ],
        'how_to_use': [
            'Apply sandalwood paste on skin',
            'Sandalwood powder with water for digestion',
            'Use in face packs for skin glow',
            'Sandalwood oil for aromatherapy'
        ],
        'precautions': [
            'Generally safe for external use',
            'Internal use should be under guidance',
            'May cause allergies in some individuals',
            'Use authentic red sandalwood only'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Bitter, Sweet',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'rose': {
        'common_name': 'Rose (Gulab)',
        'scientific_name': 'Rosa species',
        'hindi_name': 'गुलाब',
        'family': 'Rosaceae',
        'description': 'Rose is not just a beautiful flower but also has significant therapeutic and medicinal values for skin and mental health.',
        'medicinal_uses': [
            'Soothes skin irritations and inflammations',
            'Relieves stress and anxiety',
            'Improves digestion and appetite',
            'Natural astringent for skin',
            'Antidepressant properties'
        ],
        'health_benefits': [
            'Antioxidant properties for skin health',
            'Antibacterial and antiviral effects',
            'Calms nervous system',
            'Improves mood and reduces stress',
            'Anti-inflammatory properties'
        ],
        'how_to_use': [
            'Use rose water as skin toner',
            'Drink rose tea for relaxation',
            'Apply rose petal paste on skin',
            'Use rose essential oil for aromatherapy'
        ],
        'precautions': [
            'Generally safe for most people',
            'Some may be allergic to rose pollen',
            'Use organic roses for consumption',
            'Essential oil should be properly diluted'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet, Astringent, Bitter',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'sapota': {
        'common_name': 'Sapodilla (Chikoo)',
        'scientific_name': 'Manilkara zapota',
        'hindi_name': 'चीकू',
        'family': 'Sapotaceae',
        'description': 'Sapota is a sweet fruit with numerous health benefits, rich in nutrients and dietary fiber.',
        'medicinal_uses': [
            'Treats constipation and digestive issues',
            'Boosts energy and prevents anemia',
            'Supports bone health',
            'Anti-inflammatory properties',
            'Improves vision health'
        ],
        'health_benefits': [
            'High in dietary fiber for digestion',
            'Rich in iron for anemia prevention',
            'Contains calcium for bone health',
            'Antioxidant properties',
            'Natural energy booster'
        ],
        'how_to_use': [
            'Eat ripe fruit as snack',
            'Use in milkshakes and desserts',
            'Apply fruit pulp on skin',
            'Leaf decoction for fever'
        ],
        'precautions': [
            'High in natural sugars - moderate if diabetic',
            'Unripe fruit may cause mouth irritation',
            'May cause allergies in sensitive individuals',
            'Consume in moderation'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sweet',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sweet',
            'Dosha': 'Balances Vata and Pitta'
        }
    },

    'tulasi': {
        'common_name': 'Tulsi (Holy Basil)',
        'scientific_name': 'Ocimum tenuiflorum',
        'hindi_name': 'तुलसी',
        'family': 'Lamiaceae',
        'description': 'Tulsi, also known as Holy Basil, is a sacred plant in Hinduism with powerful medicinal properties. Known as the "Queen of Herbs" in Ayurveda.',
        'medicinal_uses': [
            'Boosts immunity and fights infections',
            'Reduces stress and anxiety naturally',
            'Helps in respiratory disorders like asthma and bronchitis',
            'Lowers blood sugar and cholesterol levels',
            'Anti-inflammatory and pain-relieving properties'
        ],
        'health_benefits': [
            'Rich in antioxidants that fight free radicals',
            'Contains essential oils like eugenol for pain relief',
            'Adaptogenic properties help body cope with stress',
            'Antibacterial and antiviral effects',
            'Supports heart health and circulation'
        ],
        'how_to_use': [
            'Chew 2-3 fresh leaves every morning on empty stomach',
            'Make tulsi tea by boiling 5-6 leaves in water',
            'Use tulsi extract in honey for cough and cold',
            'Apply tulsi paste on skin for infections and acne'
        ],
        'precautions': [
            'Avoid during pregnancy without doctor consultation',
            'May slow blood clotting - stop before surgery',
            'Can lower blood sugar - monitor if diabetic',
            'May interact with blood thinning medications'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Pungent, Bitter',
            'Virya (Potency)': 'Heating',
            'Vipaka (Post-digestive effect)': 'Pungent',
            'Dosha': 'Balances Kapha and Vata'
        }
    },

    'wood_sorel': {
        'common_name': 'Wood Sorrel',
        'scientific_name': 'Oxalis acetosella',
        'hindi_name': 'खट्टी बूटी',
        'family': 'Oxalidaceae',
        'description': 'Wood Sorrel is a small medicinal plant with sour taste, known for its cooling and digestive properties.',
        'medicinal_uses': [
            'Fever and inflammation reduction',
            'Digestive issues and appetite improvement',
            'Skin conditions and wounds',
            'Urinary tract infections',
            'Mouth ulcers and sore throat'
        ],
        'health_benefits': [
            'Rich in Vitamin C',
            'Antioxidant properties',
            'Anti-inflammatory effects',
            'Diuretic properties',
            'Cooling effect on body'
        ],
        'how_to_use': [
            'Chew leaves for digestive issues',
            'Leaf juice for fever',
            'Paste application for skin',
            'Herbal tea for urinary problems'
        ],
        'precautions': [
            'Contains oxalic acid - avoid in large quantities',
            'May interact with kidney medications',
            'Not recommended for people with kidney stones',
            'Use in moderation'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Sour',
            'Virya (Potency)': 'Cooling',
            'Vipaka (Post-digestive effect)': 'Sour',
            'Dosha': 'Balances Pitta and Kapha'
        }
    },

    'unknown': {
        'common_name': 'Unknown Plant',
        'scientific_name': 'Unidentified Species',
        'hindi_name': 'अपरिचित पौधा',
        'family': 'Unknown Family',
        'description': 'This plant could not be identified with sufficient confidence. It may not be in our medicinal plants database or the image quality may be insufficient.',
        'medicinal_uses': [
            'Cannot recommend medicinal uses for unidentified plants',
            'Consult with a botanist or Ayurvedic expert',
            'Proper identification is essential for safe usage'
        ],
        'health_benefits': [
            'Unknown - requires proper identification',
            'Some plants may be toxic if misidentified',
            'Always verify with experts before use'
        ],
        'how_to_use': [
            'DO NOT USE until properly identified',
            'Consult local botanical garden or expert',
            'Take clear photos from multiple angles for identification'
        ],
        'precautions': [
            '⚠️ DO NOT CONSUME unidentified plants',
            '⚠️ Some plants can be toxic or poisonous',
            '⚠️ Always verify with multiple sources',
            '⚠️ Consult qualified Ayurvedic practitioner'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Unknown',
            'Virya (Potency)': 'Unknown',
            'Vipaka (Post-digestive effect)': 'Unknown',
            'Dosha': 'Unknown - Requires proper identification'
        }
    }
}

def load_model():
    """Load trained model and metadata"""
    global model, scaler, class_mapping, training_info
    
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("feature_scaler.pkl")
        class_mapping = joblib.load("class_mapping.pkl")
        training_info = joblib.load("training_info.pkl")
        
        logger.info("✅ Enhanced model loaded successfully")
        logger.info(f"📊 Model trained on: {training_info.get('timestamp', 'Unknown')}")
        logger.info(f"📈 Test accuracy: {training_info.get('test_accuracy', 'Unknown'):.4f}")
        logger.info(f"🏷️ Number of classes: {len(class_mapping)}")
        logger.info(f"🔍 Available classes: {list(class_mapping.values())}")
        logger.info(f"🔧 Feature length: {training_info.get('feature_length', 'Unknown')}")
        
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not load model: {e}")
        return False

def calculate_skewness(data):
    """Calculate skewness manually"""
    if len(data) == 0 or np.std(data) == 0:
        return 0.0
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data):
    """Calculate kurtosis manually"""
    if len(data) == 0 or np.std(data) == 0:
        return 0.0
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 4) - 3

def enhanced_extract_features(image_path):
    """Enhanced feature extraction - EXACTLY MATCHING TRAINING FEATURES"""
    try:
        # Read and resize image to HIGH resolution
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"❌ Could not read image: {image_path}")
            return None
            
        # HIGH RESOLUTION for maximum accuracy - MUST MATCH TRAINING
        resized = cv2.resize(image, (512, 512))
        
        features = []
        
        # ===== ENHANCED COLOR FEATURES (60 features) =====
        
        # BGR color space - 10 features per channel
        for channel in range(3):
            channel_data = resized[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.percentile(channel_data, 10),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                np.percentile(channel_data, 90),
                np.var(channel_data)
            ])  # 10 × 3 = 30 features
        
        # HSV color space - 8 features per channel
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        for channel in range(3):
            channel_data = hsv[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                calculate_skewness(channel_data),
                calculate_kurtosis(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                len(channel_data)  # Additional feature
            ])  # 8 × 3 = 24 features
        
        # LAB color space - 4 features per channel
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        for channel in range(3):
            channel_data = lab[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.var(channel_data)
            ])  # 4 × 3 = 12 features
        
        # Additional color moments
        for channel in range(3):
            channel_data = resized[:, :, channel].flatten()
            if len(channel_data) > 0:
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                skew_val = calculate_skewness(channel_data)
                features.extend([mean, std, skew_val])
            else:
                features.extend([0, 0, 0])
        # 3 moments × 3 channels = 9 features
        
        # Total color features: 30 + 24 + 12 + 9 = 75 features
        
        # ===== ENHANCED TEXTURE FEATURES (70 features) =====
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Multiple Sobel gradients with different kernels
        sobel_kernels = [3, 5]
        for ksize in sobel_kernels:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            features.extend([
                np.mean(sobelx), np.std(sobelx), np.median(sobelx),
                np.mean(sobely), np.std(sobely), np.median(sobely),
                np.mean(sobel_magnitude), np.std(sobel_magnitude)
            ])  # 8 features × 2 kernels = 16 features
        
        # Multiple Laplacian filters
        laplacian_kernels = [3, 5]
        for ksize in laplacian_kernels:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
            features.extend([
                np.mean(laplacian), np.std(laplacian), np.median(laplacian),
                np.var(laplacian)
            ])  # 4 features × 2 kernels = 8 features
        
        # Enhanced Gabor filters with multiple parameters
        gabor_params = [
            (5.0, np.pi/4, 10.0, 0.5),   # theta, lambda, sigma, gamma
            (3.0, np.pi/2, 8.0, 0.8),
            (7.0, np.pi/6, 12.0, 0.3)
        ]
        for theta, lambd, sigma, gamma in gabor_params:
            gabor_kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            gabor_filtered = cv2.filter2D(gray, cv2.CV_64F, gabor_kernel)
            features.extend([
                np.mean(gabor_filtered), np.std(gabor_filtered),
                np.median(gabor_filtered), np.var(gabor_filtered)
            ])  # 4 features × 3 filters = 12 features
        
        # Canny edges with multiple thresholds
        canny_thresholds = [(50, 150), (100, 200)]
        for thresh1, thresh2 in canny_thresholds:
            edges = cv2.Canny(gray, thresh1, thresh2)
            features.extend([
                np.mean(edges), np.std(edges), 
                np.sum(edges > 0) / edges.size,  # Edge density
                np.var(edges)
            ])  # 4 features × 2 thresholds = 8 features
        
        # Local Binary Pattern-like features
        lbp_features = compute_lbp_features(gray)
        features.extend(lbp_features)  # 8 features
        
        # Gray level co-occurrence matrix (GLCM) like features
        glcm_features = compute_glcm_features(gray)
        features.extend(glcm_features)  # 6 features
        
        # Additional texture statistics
        gray_flat = gray.flatten()
        features.extend([
            np.mean(gray), np.std(gray), np.median(gray), np.var(gray),
            calculate_skewness(gray_flat),
            calculate_kurtosis(gray_flat),
            np.percentile(gray, 10), np.percentile(gray, 90),
            (np.percentile(gray, 75) - np.percentile(gray, 25))  # IQR
        ])  # 9 features
        
        # Total texture features: 16 + 8 + 12 + 8 + 8 + 6 + 9 = 67 features
        
        # ===== ENHANCED SHAPE FEATURES (40 features) =====
        shape_features = []
        
        # Multiple threshold levels for robust shape detection
        thresholds = [100, 127, 150]
        for threshold in thresholds:
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                
                if perimeter > 0:
                    shape_features.extend([
                        area, 
                        perimeter, 
                        area/perimeter,  # Compactness
                        hull_area/area if area > 0 else 0,  # Solidity
                        4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0,  # Circularity
                        len(largest_contour)  # Contour points
                    ])
                else:
                    shape_features.extend([0, 0, 0, 0, 0, 0])
            else:
                shape_features.extend([0, 0, 0, 0, 0, 0])
        
        # Shape features: 6 features × 3 thresholds = 18 features
        
        # Additional shape features
        shape_features.extend([
            gray.shape[1] / gray.shape[0],  # Aspect ratio
            np.sum(thresh > 0) / thresh.size if 'thresh' in locals() else 0,  # Foreground ratio
        ])  # 2 features
        
        # Total shape features: 18 + 2 = 20 features
        
        # ===== ENHANCED HISTOGRAM FEATURES (35 features) =====
        hist_features = []
        
        # Enhanced color histograms
        for channel in range(3):
            hist = cv2.calcHist([resized], [channel], None, [32], [0, 256])
            hist_features.extend([
                np.mean(hist), 
                np.std(hist),
                np.median(hist),
                np.max(hist),
                np.argmax(hist),  # Dominant color bin
                np.sum(hist > np.mean(hist)) / len(hist)  # Above-average ratio
            ])  # 6 features × 3 channels = 18 features
        
        # Gray histogram with enhanced features
        gray_hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist_features.extend([
            np.mean(gray_hist), 
            np.std(gray_hist),
            np.median(gray_hist),
            np.max(gray_hist),
            np.argmax(gray_hist),
            np.sum(gray_hist > 0.1 * np.max(gray_hist)) / len(gray_hist),
        ])  # 6 features
        
        # Histogram entropy
        hist_normalized = gray_hist / np.sum(gray_hist)
        hist_normalized = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized))
        hist_features.append(entropy)  # 1 feature
        
        # Total histogram features: 18 + 6 + 1 = 25 features
        
        # ===== COMBINE ALL FEATURES =====
        all_features = np.array(features + shape_features + hist_features)
        
        # Handle NaN values
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"🔍 Extracted {len(all_features)} features for prediction")
        
        # Get expected feature length from training
        expected_length = training_info.get('feature_length', 187)
        
        # If we have fewer features, pad with zeros
        if len(all_features) < expected_length:
            logger.warning(f"⚠️ Feature mismatch: Got {len(all_features)}, expected {expected_length}. Padding with zeros.")
            all_features = np.pad(all_features, (0, expected_length - len(all_features)), 'constant')
        # If we have more features, truncate
        elif len(all_features) > expected_length:
            logger.warning(f"⚠️ Feature mismatch: Got {len(all_features)}, expected {expected_length}. Truncating.")
            all_features = all_features[:expected_length]
        
        return all_features
        
    except Exception as e:
        logger.error(f"❌ Error extracting features: {e}")
        return None

def compute_lbp_features(gray):
    """Compute Local Binary Pattern features"""
    try:
        lbp_features = []
        height, width = gray.shape
        
        # Simplified LBP computation
        patterns = []
        for i in range(1, height-1, 4):  # Sample every 4th pixel for speed
            for j in range(1, width-1, 4):
                center = gray[i,j]
                binary_pattern = 0
                neighbors = [
                    gray[i-1,j-1], gray[i-1,j], gray[i-1,j+1],
                    gray[i,j+1], gray[i+1,j+1], gray[i+1,j],
                    gray[i+1,j-1], gray[i,j-1]
                ]
                
                for idx, neighbor in enumerate(neighbors):
                    if neighbor > center:
                        binary_pattern |= (1 << (7-idx))
                
                patterns.append(binary_pattern)
        
        if patterns:
            return [
                np.mean(patterns),
                np.std(patterns),
                np.median(patterns),
                np.var(patterns),
                np.percentile(patterns, 25),
                np.percentile(patterns, 75),
                len(set(patterns)) / len(patterns) if patterns else 0,  # Unique patterns ratio
                np.sum(np.array(patterns) > 127) / len(patterns) if patterns else 0,  # High value ratio
            ]
        else:
            return [0] * 8
    except:
        return [0] * 8

def compute_glcm_features(gray):
    """Compute GLCM-like texture features"""
    try:
        # Simplified GLCM computation
        diff_x = gray[1:, :] - gray[:-1, :]  # Horizontal differences
        diff_y = gray[:, 1:] - gray[:, :-1]  # Vertical differences
        
        return [
            np.mean(np.abs(diff_x)),
            np.std(diff_x),
            np.mean(np.abs(diff_y)),
            np.std(diff_y),
            np.mean(diff_x),
            np.mean(diff_y),
        ]
    except:
        return [0] * 6

def extract_image_features(image_path):
    """Main feature extraction function"""
    return enhanced_extract_features(image_path)

def get_confidence_score(features_scaled):
    """Calculate confidence score from prediction probabilities"""
    try:
        if hasattr(model, 'predict_proba') and features_scaled is not None:
            probabilities = model.predict_proba(features_scaled)
            max_prob = np.max(probabilities[0])
            confidence = float(max_prob * 100)
            
            if confidence < 50:
                return confidence, "Very Low Health"
            elif confidence < 70:
                return confidence, "Low Health"
            elif confidence < 85:
                return confidence, "Medium Health"
            else:
                return confidence, "High Health"
        return 50.0, "Low Health"
    except Exception as e:
        logger.error(f"❌ Error calculating confidence: {e}")
        return 45.0, "Very Low Health"

def get_medicinal_info(predicted_class, confidence):
    """Get medicinal plant information based on predicted class with confidence check"""
    
    # If confidence is too low, return unknown plant
    if confidence < 60:  # Increased threshold for better accuracy
        return medicinal_plants_database['unknown'], "Unknown Plant"
    
    # Convert to lowercase for matching
    predicted_lower = predicted_class.lower().replace(' ', '_')
    
    # Try exact match
    if predicted_lower in medicinal_plants_database:
        return medicinal_plants_database[predicted_lower], predicted_class
    
    # Try partial matching with higher threshold
    for plant_key, plant_info in medicinal_plants_database.items():
        if plant_key == 'unknown':
            continue
            
        # Remove underscores and spaces for better matching
        clean_predicted = predicted_lower.replace('_', '').replace(' ', '')
        clean_plant_key = plant_key.replace('_', '').replace(' ', '')
        clean_common_name = plant_info['common_name'].lower().replace(' ', '')
        
        if (clean_plant_key in clean_predicted or 
            clean_predicted in clean_plant_key or
            clean_common_name in clean_predicted or
            clean_predicted in clean_common_name):
            return plant_info, plant_info['common_name']
    
    # Try matching with scientific name
    for plant_key, plant_info in medicinal_plants_database.items():
        if plant_key == 'unknown':
            continue
            
        scientific_lower = plant_info['scientific_name'].lower()
        if predicted_lower in scientific_lower or scientific_lower in predicted_lower:
            return plant_info, plant_info['common_name']
    
    # Return unknown if no good match found
    return medicinal_plants_database['unknown'], "Unknown Plant"

# Initialize model
load_model()

# Create uploads directory
if not os.path.exists('./uploads'):
    os.makedirs('./uploads')

# Flask Routes
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """Predict plant from image using enhanced features"""
    try:
        if request.method == 'POST' and 'image' in request.files:
            f = request.files['image']
            
            if f.filename == '':
                return jsonify({'message': 'No file selected', 'status': 400})
            
            filename = werkzeug.utils.secure_filename(f.filename)
            file_path = os.path.join('./uploads', filename)
            f.save(file_path)
            
            # Validate file
            if not os.path.exists(file_path):
                return jsonify({'message': 'File upload failed', 'status': 400})
            
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                os.remove(file_path)
                return jsonify({'message': 'File too large. Maximum size is 10MB.', 'status': 400})
            
            if model is None or scaler is None:
                os.remove(file_path)
                return jsonify({
                    'message': 'Model not loaded. Please train the model first.',
                    'status': 400
                })
            
            # Extract features
            features = extract_image_features(file_path)
            if features is None:
                os.remove(file_path)
                return jsonify({'message': 'Could not process image', 'status': 400})
            
            # Ensure feature length matches scaler expectation
            expected_features = training_info.get('feature_length', 187)
            if len(features) != expected_features:
                logger.warning(f"⚠️ Feature length mismatch: {len(features)} vs {expected_features}")
                if len(features) < expected_features:
                    features = np.pad(features, (0, expected_features - len(features)), 'constant')
                else:
                    features = features[:expected_features]
            
            # Scale features
            try:
                features_scaled = scaler.transform([features])
            except Exception as e:
                logger.error(f"❌ Error scaling features: {e}")
                os.remove(file_path)
                return jsonify({'message': 'Error processing image features', 'status': 400})
            
            # Make prediction
            predicted_class_idx = model.predict(features_scaled)[0]
            predicted_class_name = class_mapping.get(predicted_class_idx, 'Unknown Plant')
            
            # Get confidence
            confidence, confidence_level = get_confidence_score(features_scaled)
            
            # Get medicinal information WITH CONFIDENCE CHECK
            medicinal_info, display_name = get_medicinal_info(predicted_class_name, confidence)
            
            # Prepare comprehensive response
            response = {
                'filename': filename,
                'message': 'Plant identification completed successfully',
                
                'plant_identification': {
                    'detected_class': predicted_class_name,
                    'display_name': display_name,
                    'common_name': medicinal_info['common_name'],
                    'scientific_name': medicinal_info['scientific_name'],
                    'hindi_name': medicinal_info['hindi_name'],
                    'family': medicinal_info['family'],
                    'description': medicinal_info['description'],
                    'is_unknown': medicinal_info['common_name'] == 'Unknown Plant'
                },
                
                'medicinal_properties': {
                    'medicinal_uses': medicinal_info['medicinal_uses'],
                    'health_benefits': medicinal_info['health_benefits'],
                    'how_to_use': medicinal_info['how_to_use'],
                    'precautions': medicinal_info['precautions']
                },
                
                'ayurvedic_properties': medicinal_info['ayurvedic_properties'],
                
                'confidence': {
                    'score': f"{confidence:.2f}%",
                    'level': confidence_level,
                    'raw_score': confidence
                },
                
                'model_info': {
                    'feature_count': len(features),
                    'model_type': training_info.get('model_type', 'Random Forest'),
                    'training_date': training_info.get('timestamp', 'Unknown'),
                    'total_classes': len(class_mapping)
                },
                
                'timestamp': datetime.now().isoformat(),
                'status': 200
            }
            
            if medicinal_info['common_name'] == 'Unknown Plant':
                logger.warning(f"⚠️ LOW CONFIDENCE Prediction: {predicted_class_name} (Confidence: {confidence:.2f}%) - Marked as Unknown")
            else:
                logger.info(f"✅ Enhanced Prediction: {display_name} (Confidence: {confidence:.2f}%)")
            
            logger.info(f"📊 Features used: {len(features)}")
            
            return jsonify(response)
        else:
            return jsonify({'message': 'No image file provided', 'status': 400})
            
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        return jsonify({'message': f'Error during prediction: {str(e)}', 'status': 500})
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

@app.route('/train-model', methods=['POST'])
@cross_origin()
def train_model():
    """Trigger model training"""
    return jsonify({
        'status': 200,
        'message': 'Please run train_model.py separately for training'
    })

@app.route('/model-info', methods=['GET'])
@cross_origin()
def model_info():
    """Get model information"""
    return jsonify({
        'status': 200,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'training_info': training_info,
        'available_classes': list(class_mapping.values()) if class_mapping else [],
        'total_medicinal_plants': len(medicinal_plants_database),
        'feature_length': training_info.get('feature_length', 0) if training_info else 0
    })

@app.route('/medicinal-plants', methods=['GET'])
@cross_origin()
def get_all_medicinal_plants():
    """Get all medicinal plants information"""
    # Convert to frontend-friendly format
    plants_list = []
    for key, info in medicinal_plants_database.items():
        if key != 'unknown':  # Exclude unknown from list
            plants_list.append({
                'key': key,
                'common_name': info['common_name'],
                'scientific_name': info['scientific_name'],
                'hindi_name': info['hindi_name'],
                'family': info['family'],
                'description': info['description']
            })
    
    return jsonify({
        'status': 200,
        'total_plants': len(plants_list),
        'medicinal_plants': plants_list
    })

@app.route('/plant-details/<plant_key>', methods=['GET'])
@cross_origin()
def get_plant_details(plant_key):
    """Get detailed information about specific plant"""
    plant_info = medicinal_plants_database.get(plant_key.lower())
    if plant_info:
        return jsonify({
            'status': 200,
            'plant': plant_info
        })
    else:
        return jsonify({
            'status': 404,
            'message': 'Plant not found'
        })

@app.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'available_classes': len(class_mapping),
        'feature_length': training_info.get('feature_length', 0) if training_info else 0,
        'server_time': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🌿 40 MEDICINAL PLANTS IDENTIFIER API")
    print("=" * 50)
    
    if model and scaler:
        print(f"✅ Enhanced model loaded successfully")
        print(f"📊 Trained on: {training_info.get('timestamp', 'Unknown')}")
        print(f"📈 Test Accuracy: {training_info.get('test_accuracy', 0):.2%}")
        print(f"🏷️ Classes: {len(class_mapping)}")
        print(f"🔧 Features per image: {training_info.get('feature_length', 0)}")
        print(f"💊 Medicinal Plants Database: {len(medicinal_plants_database)} plants")
        print(f"🛡️ Unknown Plant Detection: ENABLED")
    else:
        print("⚠️ No model loaded. Please train the enhanced model first.")
        print("💡 Run: python train_model.py")
    
    print("\n🌐 Frontend: http://localhost:4000")
    print("🔗 API Endpoints:")
    print("   - POST /predict          - Identify plant from image")
    print("   - GET  /model-info       - Model information") 
    print("   - GET  /medicinal-plants - All plants list")
    print("   - GET  /plant-details/*  - Specific plant details")
    print("   - GET  /health           - Health check")
    print(f"\n🌿 Available Plants: {len(medicinal_plants_database)-1} medicinal plants")
    
    app.run(host="0.0.0.0", port=4000, debug=True)