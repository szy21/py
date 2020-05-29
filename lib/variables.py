var_name_dict_AM4 = {
    'olr':'lwup_toa','olr_clr':'lwup_toa_clr',
    'LWP':'lwp','IWP':'iwp',
    'aer_ex_c_vs':'aod','aer_ab_c_vs':'aod_ab',
    'low_cld_amt':'low_cld','mid_cld_amt':'mid_cld','high_cld_amt':'high_cld','tot_cld_amt':'tot_cld',
    'reff_modis':'reff_modis_sum', 'reff_modis2':'reff_modis_ct',
    'tasmax':'t_ref_max'
    }

var_name_dict_CAM6 = {
    'SWCF':'swcre','LWCF':'lwcre',
    'FLNT':'lwnet_toa','FLNTC':'lwnet_toa_clr','FLUT':'lwup_toa','FLUTC':'lwup_toa_clr','FSNT':'swnet_toa','FSNTC':'swnet_toa_clr',
    'FLDS':'lwdn_sfc','FLNS':'lwnet_sfc','FLNSC':'lwnet_sfc_clr','FSDS':'swdn_sfc','FSDSC':'swdn_sfc_clr','FSNS':'swnet_sfc','FSNSC':'swnet_sfc_clr',
    'PRECC':'prec_conv','PRECL':'prec_ls','TS':'t_surf','TSMX':'t_surf_max','TREFHT':'t_ref','AEROD_v':'aod',
    'CLDLOW':'low_cld','CLDMED':'mid_cld','CLDHGH':'high_cld','CLDTOT':'tot_cld',
    'CLDLIQ':'liq_wat',
    'TGCLDLWP':'lwp','TGCLDIWP':'iwp','TGCLDCWP':'cwp','AREL':'arel','AWNC':'cdnc','CCN3':'ccn3','FREQL':'freql'
    }

unit_dict_CAM6 = {
    'CLDLOW':100.0,'CLDMID':100.0,'CLDHGH':100.0,'CLDTOT':100.0,
    'PRECC':86400.0,'PRECL':86400.0
    }
