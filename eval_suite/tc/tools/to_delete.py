def quantile_probabilities_pf(input_field, bin_values, day, member, time):
    values, bins = np.histogram(
        input_field[day, member, time, :, :].flatten(), bin_values
    )
    bins_prob = values / float(
        (input_field[day, member, time, :, :].flatten()).shape[0]
    )
    bins_mid = (bins[1:] + bins[:-1]) / 2
    bins_width = bins[1] - bins[0]
    return (bins_prob, bins_mid)


def quantile_probabilities_cf(input_field, bin_values, day, time):
    values, bins = np.histogram(input_field[day, time, :, :].flatten(), bin_values)
    bins_prob = values / float((input_field[day, time, :, :].flatten()).shape[0])
    bins_mid = (bins[1:] + bins[:-1]) / 2
    bins_width = bins[1] - bins[0]
    return (bins_prob, bins_mid)


def quantile_probabilities_left10(input_field, bin_values, day, member, time):
    data = input_field[day, member, time].flatten()
    thresh = np.nanquantile(data, 0.01)
    data = data[data < thresh]
    values, bins = np.histogram(data, bin_values)
    bins_prob = values / data.shape[0]
    bins_mid = (bins[1:] + bins[:-1]) / 2
    return bins_prob, bins_mid


def compute_probabilities(data, bin_values, day, time, member=None, is_pf=True):
    if is_pf:
        return quantile_probabilities_pf(data, bin_values, day, member, time)
    else:
        return quantile_probabilities_cf(data, bin_values, day, time)


def compute_probabilities_for_variable(
    variable_data, xbins, list_exps, list_expid_ml, dates, ensemble_size, time
):
    # Create a dictionary to hold bins for each dataset
    bins_prob = {}
    bins_mid = {}

    # Initialize bins for each dataset
    for exp in list_exps:
        if exp == "OPER_O320_0001":
            bins_prob[exp] = np.zeros((len(dates), len(xbins) - 1))
            bins_mid[exp] = np.zeros((len(dates), len(xbins) - 1))
        else:
            bins_prob[exp] = np.zeros((len(dates), ensemble_size, len(xbins) - 1))
            bins_mid[exp] = np.zeros((len(dates), ensemble_size, len(xbins) - 1))

    # Compute frequency distributions for each day and ensemble member
    for day in range(len(dates)):
        for member in range(ensemble_size):
            for key in ["EEFO_O96_0001", "EEFO_O320_0001", "ENFO_O320_0001"]:
                (
                    bins_prob[key][day, member, :],
                    bins_mid[key][day, member, :],
                ) = compute_probabilities(variable_data[key], xbins, day, time, member)

            for expid in list_expid_ml:
                (
                    bins_prob[expid][day, member, :],
                    bins_mid[expid][day, member, :],
                ) = compute_probabilities(
                    variable_data[expid], xbins, day, time, member
                )

        (
            bins_prob["OPER_O320_0001"][day, :],
            bins_mid["OPER_O320_0001"][day, :],
        ) = compute_probabilities(
            variable_data["OPER_O320_0001"], xbins, day, time, is_pf=False
        )

    # Average distributions across all ensemble members
    mean_prob = {key: np.nanmean(bins_prob[key], axis=1) for key in bins_prob.keys()}

    return {
        "bins_mid": bins_mid,
        "bins_prob": bins_prob,
        "mean_prob": mean_prob,
    }


def compute_wind10m_probabilities(
    wind10m, list_exps, list_expid_ml, dates, ensemble_size, time
):
    xbins_wind10m = np.arange(0, 30.01, 1)
    return compute_probabilities_for_variable(
        wind10m, xbins_wind10m, list_exps, list_expid_ml, dates, ensemble_size, time
    )


def compute_msl_probabilities(
    msl, list_exps, list_expid_ml, dates, ensemble_size, time
):
    xbins_msl = np.arange(960, 1020, 1)
    return compute_probabilities_for_variable(
        msl, xbins_msl, list_exps, list_expid_ml, dates, ensemble_size, time
    )
