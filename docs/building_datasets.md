# Building the datasets


Denver FIPS = `08031`


## Types of data

#### - multi-region without targets

A sample of the descriptive stats for houses for clustering, categorization techniques

Using sampling methods

#### - single region, with targets

Can be used for cross validation



## Dataset

Denver prediction dataset

```sql
select pd.*, target
from target_rec t
left join prediction_dataset pd
    using (RTPropertyUniqueIdentifier)
where t.SitusStateCountyFIPS in (
    select fips from investor_region_exploded_view where region_name='CO-Denver')
-- limit 10;
;
```


```sql
update table target_random_id_fips_rt t
inner join realtytrac_tax tax
    using (RTPropertyUniqueIdentifier)
set t.fips = tax.SitusStateCountyFIPS;
```

Get 10% for the PCA and clustering

```sql
drop table tmp_ds_cluster;

create table tmp_ds_cluster as
select r.*, tax.OwnerOccupiedInt, 
  tax.YearBuilt, 
  tax.SquareFootage, tax.LotSize, 
  tax.EstimatedValue,
  tax.TaxAssessedValue, tax.TaxImprovementValue, tax.TaxLandValue, tax.TaxImprovementPercent,
  tax.TaxAmount
from target_random_id_fips_rt r
left join realtytrac_tax tax
  on r.id=tax.RTPropertyUniqueIdentifier
where random < .2
and fips in (select distinct fips from investor_region_exploded_view)
and tax.SFR=1
-- limit 10;
;
```

Updates

```sql
update tmp_ds_cluster
set TaxImprovementPercent = 100 * TaxImprovementValue / TaxAssessedValue;

update tmp_ds_cluster
set TaxImprovementPercent = 100 where TaxImprovementPercent>100;

update tmp_ds_cluster
set TaxImprovementPercent = 0 where TaxImprovementPercent is null;
```



