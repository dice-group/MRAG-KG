import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1


evaluation_samples = {
    "http://example.org/image_1082": "What outfits feature black shorts ?",
    "http://example.org/image_42527": "What are some jumpsuits with a high waist ?",
    "http://example.org/image_39373": "What are some dress options that feature a round neckline ?",
    "http://example.org/image_34524": "What are some short-sleeved tops ?",
    "http://example.org/image_38547": "What are some fashionable dress options with a high waist ?",
    "http://example.org/image_21806": "What casual dress styles with short sleeves and round necklines are available for warm weather ?",
    "http://example.org/image_41097": "What formal dresses have a high-low silhouette ?",
    "http://example.org/image_5434": "Featuring a black sleeveless top with a scoop neckline, what would you recommend ?",
    "http://example.org/image_889": "What are some outerwear options that feature a fur collar ?",
    "http://example.org/image_48024": "What are some light pink knitted sweaters ?",
    "http://example.org/image_23951": "What are some fashionable outfits that include a red skirt that reaches just above the knees ?",
    "http://example.org/image_34371": "Can you recommend any dresses with a heavy, possibly satin or silk fabric ?",
    "http://example.org/image_26553": "What are some fashionable dress options with a floral print ?",
    "http://example.org/image_39479": "What are some clothing options that include a black tank top with a unique graphic design on the front ?",
    "http://example.org/image_17925": "What are some outfits that include a black mesh skirt ?",
    "http://example.org/image_24171": "What are some dresses with a fitted bodice ?",
    "http://example.org/image_6829": "What casual and comfortable outfit features blue denim jeans with a faded wash ?",
    "http://example.org/image_4014": "What are some dress shirts without any special manufacturing techniques or non-textile materials ?",
    "http://example.org/image_36802": "What are some clothing options that include a watch on the left wrist ?",
    "http://example.org/image_4174": "What fashionable ensemble would you recommend for a high-waisted, gold embellished skirt ?",
    "http://example.org/image_30775": "What are some fashionable outfit options for a slim-fit light blue suit ?",
    "http://example.org/image_32118": "What is a leopard print blouse with a high neckline ?",
    "http://example.org/image_39494": "What casual yet stylish outfits would you recommend featuring orange pants with a straight cut and a flat front ?",
    "http://example.org/image_1069": "Dark blue jeans, what would you recommend ?",
    "http://example.org/image_30081": "What light blue button-up shirts are available ?",
    "http://example.org/image_14081": "What are some outfits featuring sandals with a strap across the foot ?",
    "http://example.org/image_8082": "What are some leather outfits with a belted waist ?",
    "http://example.org/image_21051": "What are some outfits that include a black belt ?",
    "http://example.org/image_49508": "What are some long-sleeved sweatshirts ?",
    "http://example.org/image_14325": "What are some jumpsuits with symmetrical silhouette ?",
    "http://example.org/image_1352": "What are some dress options suitable for daytime events ?",
    "http://example.org/image_81": "What are some outfits that include a black leather corset with a lace-up front ?",
    "http://example.org/image_6051": "What are some clothing options that include black high heels ?",
    "http://example.org/image_37671": "What are some clothing options that include a single-breasted blazer ?",
    "http://example.org/image_15166": "Dark shoes or boots, what would you recommend ?",
    "http://example.org/image_14193": "What are some wholebody dresses with a sequin finish ?",
    "http://example.org/image_49546": "What are some upperbody garments with a plain pattern ?",
    "http://example.org/image_12045": "What are some outfits suitable for a fashion show or similar event ?",
    "http://example.org/image_6720": "What fashion ensemble would you recommend for a formal event ?",
    "http://example.org/image_33263": "What are some wholebody dresses with a classic and elegant style ?",
    "http://example.org/image_2460": "What are some clothing options that have a slash pocket ?",
    "http://example.org/image_26441": "What casual hoodies are available ?",
    "http://example.org/image_20880": "What are some fashionable and coordinated outfits that include a black handbag ?",
    "http://example.org/image_2418": "High-heeled sandals with a strappy design ?",
    "http://example.org/image_47279": "What casual and comfortable outfits include a bright red long-sleeved sweater ?",
    "http://example.org/image_45305": "What are some edgy t-shirts ?",
    "http://example.org/image_27640": "What are some t-shirts with a floral print ?",
    "http://example.org/image_19870": "What are some attire options that pair with a dark shirt and brown leather shoes ?",
    "http://example.org/image_11359": "What are some fashionable outfits that include a black skirt that reaches just above the knee ?",
    "http://example.org/image_276": "What are some options made of a smooth fabric ?",
    "http://example.org/image_30229": "What casual spring or summer outfits would you recommend for a person with a set-in sleeve ?",
    "http://example.org/image_967": "What type of casual athletic outfit would be suitable for a workout setting ?",
    "http://example.org/image_26357": "What are some tops made of lightweight fabric ?",
    "http://example.org/image_2535": "What casual, urban-style outfits would you recommend for a relaxed setting ?",
    "http://example.org/image_11368": "What are some casual and relaxed outfits suitable for a warm day outdoors ?",
    "http://example.org/image_21672": "What type of formal wear dresses would you recommend for events like weddings, galas, or high-profile social gatherings ?",
    "http://example.org/image_10979": "What are some jumpsuits with a bold, patterned design ?",
    "http://example.org/image_37384": "What are some evening gowns with a fitted bodice ?",
    "http://example.org/image_204": "What are some outfits featuring black high-waisted pants with a fly opening and peg silhouette ?",
    "http://example.org/image_12907": "What are some outfit recommendations that include black shorts with a smooth finish ?",
    "http://example.org/image_48528": "What traditional-style, long-sleeved jackets are available online ?",
    "http://example.org/image_34867": "What dresses could be paired with black high heels ?",
    "http://example.org/image_32243": "What casual, summery outfits with a polka dot pattern would you recommend for a warm day out ?",
    "http://example.org/image_29033": "What formal or evening gowns have a fitted bodice with a sweetheart neckline ?",
    "http://example.org/image_31567": "What are some navy blue cardigans with a V-neckline ?",
    "http://example.org/image_11241": "What casual, summery outfits with a flared skirt and a fitted top would complement a red headband ?",
    "http://example.org/image_32496": "What formal or semi-formal dresses with a sleeveless design would you recommend for an occasion ?",
    "http://example.org/image_36098": "What are some formal dress options that feature a deep navy color ?",
    "http://example.org/image_9747": "What type of sneakers would you recommend for a casual, summery outfit ?",
    "http://example.org/image_42949": "What are some black jackets with a classic design ?",
    "http://example.org/image_31097": "What are some dress options made of satin or similar smooth fabric ?",
    "http://example.org/image_27229": "What are some fashionable denim jumpsuits ?",
    "http://example.org/image_26932": "What are some fashionable and stylish dress options for a formal or semi-formal event ?",
    "http://example.org/image_16304": "What color pattern would you recommend for a runway show ?",
    "http://example.org/image_40627": "What casual pants with a straight cut would you recommend ?",
    "http://example.org/image_44639": "What are some fashionable dresses with a bow neckline ?",
    "http://example.org/image_39575": "What are some dress options that are suitable for smart-casual wardrobes ?",
    "http://example.org/image_3500": "What are some clothing options that include a black and white patterned top with a round neckline ?",
    "http://example.org/image_50164": "What color trousers would complement a classic tuxedo ?",
    "http://example.org/image_6129": "What sunglasses would you recommend for a chic, bohemian look ?",
    "http://example.org/image_44775": "Can you recommend any dresses with a coral or peach color and a sparkling, embellished look ?",
    "http://example.org/image_39515": "What men's suit jackets with a welt pocket are available online ?",
    "http://example.org/image_45006": "What casual graphic t-shirts are available for babies or toddlers ?",
    "http://example.org/image_36685": "Can you recommend a beige blouse with a ruffled neckline as part of a fashionable outfit ?",
    "http://example.org/image_11571": "What are some clothing options that include a light beige blazer with a notched collar and a front zipper ?",
    "http://example.org/image_28528": "What are some professional clothing options for a single-button, notch lapel suit ?",
    "http://example.org/image_50016": "What hoodies are suitable for everyday wear or casual outings ?",
    "http://example.org/image_29005": "What are some North Face jackets ?",
    "http://example.org/image_14200": "What casual outfit would you recommend that includes a white tank top ?",
    "http://example.org/image_30547": "What are some midi length lower body pants ?",
    "http://example.org/image_43170": "What are some fashionable and trendy upperbody outfits that include a black and white faux fur vest ?",
    "http://example.org/image_49984": "What type of formal and professional attire would you recommend for a man ?",
    "http://example.org/image_45098": "What are some short-sleeved polo shirts ?",
    "http://example.org/image_50336": "What comfortable hoodies have a drawstring hood ?",
    "http://example.org/image_37321": "What are some long-sleeved tops with a crew neckline ?",
    "http://example.org/image_44497": "What are some tank tops made of cotton or cotton blend ?",
    "http://example.org/image_16047": "What sunglasses would you recommend for a pink brick wall backdrop ?",
    "http://example.org/image_22278": "Can you recommend brown suede boots ?",
    "http://example.org/image_22395": "What are some fashionable and chic coats ?",
    "http://example.org/image_15494": "What are some dresses with a cinched waist ?",
}

