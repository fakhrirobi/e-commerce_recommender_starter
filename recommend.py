def recommend_top_n(rating, n_items=10) : 
    
    rated_count = (rating.groupby('itemID',as_index=False)
                        .agg(rating_count = pd.NamedAgg('reviewerID','count'))
                        .sort_values('rating_count',ascending=False)
                        .head(n_items))
    return rated_count
recommend_top_n(rating=rating,
                n_items=30)

