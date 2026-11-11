# import core.asam as asam
import utils.asam as asam
import utils.qasam as qasam
# import utils.qasam as qasam
import utils.qsam as qsam
import utils.projected_qsam as projected_qsam
import utils.svd_projected_qsam as svd_projected_qsam
import utils.sam as sam


def get_minimizer(model, optimizer, args):
    # These checks must precede the legacy RS_QSAM/QR_QSAM substring checks.
    # In SVD_QR_QSAM, QR denotes quantization residual, not QR decomposition.
    if "SVD_RS_QSAM" in args.opt_type:
        minimizer = svd_projected_qsam.SVDRandomSubspaceQSAM(
            optimizer,
            model,
            rho=args.rho,
            rank=args.projection_rank,
            projection_seed=args.projection_seed,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "SVD_QR_QSAM" in args.opt_type:
        minimizer = svd_projected_qsam.SVDQuantizationResidualQSAM(
            optimizer,
            model,
            rho=args.rho,
            rank=args.projection_rank,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "RS_QSAM" in args.opt_type:
        minimizer = projected_qsam.RandomSubspaceQSAM(
            optimizer,
            model,
            rho=args.rho,
            rank=args.projection_rank,
            projection_seed=args.projection_seed,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "QR_QSAM" in args.opt_type:
        minimizer = projected_qsam.QuantizationResidualQSAM(
            optimizer,
            model,
            rho=args.rho,
            rank=args.projection_rank,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "QSAM" in args.opt_type:
        minimizer = qsam.QSAM(
            optimizer,
            model,
            rho=args.rho,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "QASAM" in args.opt_type:
        minimizer = qasam.QASAM(
            optimizer,
            model,
            rho=args.rho,
            eta=args.eta,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "ASAM" in args.opt_type:
        minimizer = asam.ASAM(optimizer, model, rho=args.rho, eta=args.eta)
    elif "SAM" in args.opt_type:
        SAM = sam.SAM
        minimizer = SAM(optimizer, model, rho=args.rho, eta=args.eta,)
    else:
        raise NotImplemented

    return minimizer
