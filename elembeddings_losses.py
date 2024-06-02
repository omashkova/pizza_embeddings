import torch as th
from torch.nn.functional import leaky_relu, relu


def gci0_loss(
    data,
    class_embed,
    class_rad,
    class_reg,
    margin,
    loss_type,
    neg=False,
):
    if neg:
        return gci1_bot_loss(
            data,
            class_embed,
            class_rad,
            class_reg,
            margin,
            loss_type,
        )

    else:
        loss_func = relu if loss_type == "relu" else leaky_relu
        c = class_embed(data[:, 0])
        d = class_embed(data[:, 1])
        rc = th.abs(class_rad(data[:, 0]))
        rd = th.abs(class_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        loss = loss_func(dist - margin)

        return loss + class_reg(c) + class_reg(d)


def gci0_bot_loss(data, class_rad, neg=False):
    rc = th.abs(class_rad(data[:, 0]))
    return rc


def gci1_loss(
    data,
    class_embed,
    class_rad,
    class_reg,
    margin,
    loss_type,
    neg=False,
):
    if neg:
        return gci1_loss_neg(
            data,
            class_embed,
            class_rad,
            class_reg,
            margin,
            loss_type,
        )

    else:
        loss_func = relu if loss_type == "relu" else leaky_relu
        c = class_embed(data[:, 0])
        d = class_embed(data[:, 1])
        e = class_embed(data[:, 2])
        rc = th.abs(class_rad(data[:, 0]))
        rd = th.abs(class_rad(data[:, 1]))

        sr = rc + rd

        dst = th.linalg.norm(d - c, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = (
            loss_func(dst - sr - margin)
            + loss_func(dst2 - rc - margin)
            + loss_func(dst3 - rd - margin)
        )

        return loss + class_reg(c) + class_reg(d) + class_reg(e)


def gci1_loss_neg(
    data,
    class_embed,
    class_rad,
    class_reg,
    margin,
    loss_type,
):
    loss_func = relu if loss_type == "relu" else leaky_relu
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))

    sr = rc + rd

    dst = th.linalg.norm(d - c, dim=1, keepdim=True)
    dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
    dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
    loss = (
        loss_func(dst - sr - margin)
        + loss_func(-dst2 + rc + margin)
        + loss_func(-dst3 + rd + margin)
    )

    return loss + class_reg(c) + class_reg(d) + class_reg(e)


def gci1_bot_loss(
    data,
    class_embed,
    class_rad,
    class_reg,
    margin,
    loss_type,
    neg=False,
):
    loss_func = relu if loss_type == "relu" else leaky_relu
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))

    sr = rc + rd
    dst = th.linalg.norm(d - c, dim=1, keepdim=True)
    return loss_func(sr - dst + margin) + class_reg(c) + class_reg(d)


def gci2_loss(
    data,
    class_embed,
    class_rad,
    rel_embed,
    class_reg,
    margin,
    loss_type,
    neg=False,
):
    if neg:
        return gci2_loss_neg(
            data,
            class_embed,
            class_rad,
            rel_embed,
            class_reg,
            margin,
            loss_type,
        )

    else:
        loss_func = relu if loss_type == "relu" else leaky_relu
        c = class_embed(data[:, 0])
        rE = rel_embed(data[:, 1])
        d = class_embed(data[:, 2])
        rc = th.abs(class_rad(data[:, 0]))
        rd = th.abs(class_rad(data[:, 2]))

        dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
        loss = loss_func(dst + rc - rd - margin)
        if class_reg is not None:
            return loss + class_reg(c) + class_reg(d)
        else:
            return loss


def gci2_loss_neg(
    data,
    class_embed,
    class_rad,
    rel_embed,
    class_reg,
    margin,
    loss_type,
):
    loss_func = relu if loss_type == "relu" else leaky_relu
    c = class_embed(data[:, 0])
    rE = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 2]))

    dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
    loss = loss_func(rc + rd - dst + margin)
    return loss + class_reg(c) + class_reg(d)


def gci3_loss(
    data,
    class_embed,
    class_rad,
    rel_embed,
    class_reg,
    margin,
    loss_type,
    neg=False,
):
    if neg:
        return gci3_loss_neg(
            data,
            class_embed,
            class_rad,
            rel_embed,
            class_reg,
            margin,
            loss_type,
        )

    else:
        loss_func = relu if loss_type == "relu" else leaky_relu
        rE = rel_embed(data[:, 0])
        c = class_embed(data[:, 1])
        d = class_embed(data[:, 2])
        rc = th.abs(class_rad(data[:, 1]))
        rd = th.abs(class_rad(data[:, 2]))

        euc = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
        loss = loss_func(euc + rc - rd - margin)

        return loss + class_reg(c) + class_reg(d)


def gci3_loss_neg(
    data,
    class_embed,
    class_rad,
    rel_embed,
    class_reg,
    margin,
    loss_type,
):
    loss_func = relu if loss_type == "relu" else leaky_relu
    rE = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))

    euc = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
    loss = loss_func(-euc + rc + rd + margin)

    return loss + class_reg(c) + class_reg(d)


def gci3_bot_loss(data, class_rad, neg=False):
    rc = th.abs(class_rad(data[:, 1]))
    return rc
