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

unit_dict_AM4 = {
	'low_cld_amt':0.01, 'mid_cld_amt':0.01, 'high_cld_amt':0.01, 'tot_cld_amt':0.01,
	'reff_modis':1.e-6
	}
unit_dict_CAM6 = {
    'PRECC':86400.0,'PRECL':86400.0
    }

var_name_dict_clima_to_pycles = {
    'u':'u_mean','v':'v_mean','w':'w_mean','temp':'temperature_mean',
    'qt':'qt_mean','thd':'theta_mean','thl':'thetali_mean','ql':'ql_mean','cld_frac':'cloud_fraction',
    'var_u':'u_mean2','var_v':'v_mean2','var_w':'w_mean2','w3':'w_mean3','tke':'tke_mean',
    'var_qt':'qt_mean2','var_thl':'thetali_mean2',
    'cov_w_qt':'qt_flux_z','w_qt_sgs':'qt_sgs_flux_z',
    'cld_frac':'cloud_fraction','cld_cover':'cloud_fraction','lwp':'lwp','tke_vint':'tke_vint',
    'cld_top':'cloud_top','cld_base':'cloud_base',
    'core_frac':'fraction_core','w_core':'w_core','ql_core':'ql_core','var_w_core':'w2_core'}
