import numpy as np
np.set_printoptions(threshold=np.inf)  # Just to print full size

def sequencer(dc,ph,p):
    """creates table of leg contact sequences"""
    # leg 1 on right side of segment
    boundsR = np.array([[np.remainder(np.linspace(0,(p.n-1)*ph,num=p.n),1)],
                      [np.remainder(np.linspace(dc,(p.n-1)*ph+dc,num=p.n),1)]])

    boundsL = np.array([[np.remainder(np.linspace(0+0.5,(p.n-1)*ph + 0.5,num=p.n),1)],
                      [np.remainder(np.linspace(dc+0.5,(p.n-1)*ph+dc+0.5,num=p.n),1)]])

    # contact switch times
    t = np.concatenate((boundsR[0, :], boundsL[0, :]), axis=1)
    t = np.sort(t)
    td = np.equal(np.delete(t,0) - np.delete(t,-1), 0)
    td = np.concatenate((td, [False]))
    t = t[0, np.invert(td)]
    t = t[:, None]
    t = np.kron(np.ones((1, np.ceil(p.tSpan[-1]).astype('int'))),t)
    t = np.add(np.transpose(t), np.transpose(np.linspace(0, np.size(t,1)-1, num=np.size(t,1)))[:,None])
    t = np.transpose(t).reshape(1, np.size(t, 0)*np.size(t, 1), order='F')
    t = t[None, t < p.tSpan[-1]]

    #  initialize control table
    ctrlTable = np.zeros((np.size(t, 1), 1+p.n))
    ctrlTable[:,0] = t
    t = np.remainder(t+1e-10, 1)

    #  store active contact indices to control table
    for i in range(1, np.size(t,1)+1):
    #for i in range(1, 2):
        aidxr = np.logical_or(np.logical_and(np.greater_equal(t[0,i-1], boundsR[0,:]), np.less(t[0, i-1], boundsR[1,:])),
                              np.logical_and(np.logical_or(np.greater_equal(t[0,i-1], boundsR[0,:]), np.less(t[0, i-1], boundsR[1,:])),  np.greater(boundsR[0, :], boundsR[1, :])))
        aidxl = np.logical_or(np.logical_and(np.greater_equal(t[0, i - 1], boundsL[0, :]), np.less(t[0, i - 1], boundsL[1, :])),
                              np.logical_and(np.logical_or(np.greater_equal(t[0, i - 1], boundsL[0, :]), np.less(t[0, i - 1], boundsL[1, :])), np.greater(boundsL[0, :], boundsL[1, :])))

        aidx = aidxr.astype('int') - aidxl.astype('int')
        idx = np.where(np.abs(aidx) == 1)
        idx = idx[1]
        ctrlTable[i-1, 1:idx.size+1] = np.multiply(aidx[0,idx],idx+1)

    return ctrlTable

def matrixinitializer(p):
    """initialize constant size matrices"""
    p.M0 = np.zeros((p.dim, p.dim))
    p.MInv0 = np.zeros((p.dim, p.dim))
    p.h0 = np.zeros((p.dim, 1))

    # coefficient matrix AR
    AR = np.identity(p.dim)
    AR[0:2, 0:2] = np.zeros((2,2))
    p.AR = AR

    # coefficient matrix AL
    AL = np.zeros((p.dim, p.dim))
    AL[0:2, 0:2] = np.identity(2)
    J1 = np.zeros((2, p.dim))
    J1[0:2, 0:2] = np.identity(2)
    for i in range(0, p.n):
        if np.logical_and(np.equal(i, 0), np.greater(p.n, 1)):
            J1[0,2] = 1/2
            J1[1,2] = 1/2
            J1[0,3] = 1/2
            J1[1,3] = 1/2
            AL = AL + np.dot(np.transpose(J1),J1)
        elif i > 1:
            J1[0, 1 + i] = J1[0, 1 + i] + 1/2
            J1[1, 1 + i] = J1[1, 1 + i] + 1/2
            J1[0, 2 + i] = J1[0, 2 + i] + 1/2
            J1[1, 2 + i] = J1[1, 2 + i] + 1/2
            AL = AL + np.dot(np.transpose(J1),J1)
    AL[2:, 2:] = AL[2:, 2:]/2
    p.AL = AL

    # stiffness and damping matrices
    SHat = np.zeros((p.n, p.n))
    DHat = np.zeros((p.n, p.n))
    S = np.zeros((p.n+2, p.n+2))
    D = np.zeros((p.n +2, p.n +2))
    for i in range(0, p.n):
        for j in range(0, p.n):
            if np.equal(i, j):
                if np.logical_or(np.equal(i, 0), np.equal(i, p.n-1)):
                    SHat[j,i] = -p.k
                    DHat[j,i] = -p.d
                else:
                    SHat[j, i] = -2*p.k
                    DHat[j, i] = -2*p.d
            elif np.logical_or(np.equal(i, j+1), np.equal(i, j-1)):
                SHat[j, i] = p.k
                DHat[j, i] = p.d
    for i in range(0, p.n):
        for j in range(0, p.n):
            S[2+i, 2+j] = SHat[i,j]
            D[2 + i, 2 + j] = DHat[i, j]

    p.S = S
    p.D = D
    return p

def dynamics(t, x, action, cInfo, p):
    """compute integrable dynamics"""
    x = np.reshape(x,(len(x),1))
    q = x[0:p.dim, [0]]
    u = x[p.dim:, [0]]
    [M, h, MInv, F, Jc, xi] = matrixsetup(t, q, u, action, cInfo, p)
    du = accelerations(q, u, F, h, MInv, Jc, xi, p)

    dx = np.zeros((len(q)+len(u),1))
    dx[0:p.dim, 0] = x[p.dim:2*p.dim, 0]
    dx[p.dim:2*p.dim, 0:1] = du

    return dx
    #return dx[:,0] # for ode_ivp

def contacts(t, p):
    """find index of current closed leg contacts"""
    # count from bottom element. positive indices are right body side and negative left
    idx = np.where(np.append(p.ctrlTable[:,0:1], p.tSpan[-1]) > t)
    idx = idx[0].reshape(len(idx[0]), 1)
    if idx.size > np.size(p.ctrlTable, 0):
        tEvent = p.tSpan[-1]
    else:
        tEvent = p.ctrlTable[idx[0,0], 0]  # time of next contact event

    cc = np.transpose(p.ctrlTable[idx[0,0]-1:idx[0,0], 1:])
    cc = cc[np.not_equal(cc,0)]
    cc = cc.reshape(len(cc), 1)
    alpha = np.multiply(np.sign(cc), np.ones((len(cc), 1)))*p.alpha0
    cc = abs(cc)

    return [cc, alpha, tEvent]

def contactinfo(cc, cInfo0, alpha, q, p):
    """compute contact vectors

    # cInfo encodes leg contact information: [element #, alpha, rC', rN']
    # rC active contact point vectors, rN active element CoM vectors
    """
    cc = cc.astype('int')
    if cc.size != 0:
        rc = np.zeros((np.size(cc,0), 2))
        rn = np.zeros((np.size(cc,0), 2))

        # check if contact closed or switched leg
        if cInfo0.size == 0:
            cLogic = np.zeros((p.n, 1))
        else:
            cc0 = cInfo0[:,0:1].astype('int')
            alpha0 = cInfo0[:, 1:2]
            aCmp = np.zeros((p.n, 1))
            aCmp0 = np. zeros((p.n, 1))
            aCmp[cc-1,0] = alpha
            aCmp0[cc0-1,0] = alpha0

            cLogic = np.sign(aCmp) == np.sign(aCmp0)

        sc = 1/2*np.array([-np.sin(q[2:2+p.n,0]),np.cos(q[2:2+p.n,0])])
        Jq0 = np.concatenate((np.multiply(np.identity(2), q[0:2,:]), sc), axis=1)

        # compute contact and CoM vectors
        for i in range(0, np.size(cc,0)):
            if cc[i] == 1:
                rn[i,:] = np.transpose(q[0:2,:])
            elif cc[i]>2:
                mask = np.concatenate((np.zeros((2,3)), np.ones((2, cc[i,0] -2)), np.zeros((2, p.dim - cc[i,0] -1))), axis=1)
                Jq0m = Jq0 + np.multiply(Jq0, mask)
                rn[i,:] = np.transpose(np.matmul(Jq0m[:,0:2+cc[i,0]],np.ones((2+cc[i,0], 1))))
            else:
                rn[i, :] = np.transpose(np.matmul(Jq0[:,0:2+cc[i,0]],np.ones((2+cc[i,0], 1))))
            if cLogic[cc[i]-1] != 1:
                rc[i,:] = rn[i,:] + p.g0*np.concatenate((np.sin(alpha[i] + q[cc[i]+1]), -np.cos(alpha[i] + q[cc[i]+1])), axis=1)
            else:
                rc[i,:] = cInfo0[cInfo0[:,0]==cc[i], 2:4]

        cInfo = np.concatenate((cc, alpha, rc, rn),axis=1)
    else:
        cInfo = np.array([0])
    return cInfo

def matrixsetup(t, q, u, action, cInfo, p):
    """generate all matrices"""
    # system matrices
    J = np.concatenate((np.identity(2), np.array([-np.cos(q[2:2+p.n,0]),-np.sin(q[2:2+p.n,0])])),axis=1)
    scu = np.array([np.multiply(np.sin(q[2:2+p.n,0]),u[2:2+p.n,0]),np.multiply(-np.cos(q[2:2+p.n,0]),u[2:2+p.n,0])])
    dJu = np.concatenate((np.zeros((2,2)), scu),axis=1)

    ATrig = np.matmul(np.transpose(J),J)
    ADTrig = np.multiply(p.AL, np.matmul(np.transpose(J), dJu))
    M = np.multiply(p.AL, ATrig) + p.b0*p.AR  # mass matrix
    MInv = np.linalg.inv(M)
    h = np.matmul(ADTrig, u)  # matrix of gyroscopic accelerations

    # contact dependent matrices
    F = np.zeros((p.dim, 1))

    Jc = np.zeros((np.size(cInfo,0), p.dim))
    xi = np.zeros((np.size(cInfo,0), 1))

    if np.not_equal(cInfo.size, 0):
        # contact matrices
        ncc = np.size(cInfo,0)
        rc = cInfo[:, 2:4]

        # initialize matrices
        sc = 1 / 2 * np.array([-np.sin(q[2:2 + p.n, 0]), np.cos(q[2:2 + p.n, 0])])
        Jq0 = np.concatenate((np.multiply(np.identity(2), q[0:2,:]), sc), axis=1)
        Jn0 = J
        Jn0[:,2:] = 1/2*J[:,2:]
        dJn0 = dJu
        dJn0[:,2:] = 1/2*dJu[:,2:]

        # compute contact and leg forces
        cc = cInfo[:,0:1].astype('int')
        for i in range(0, ncc):
            if cc[i, 0] ==1:
                rn = np.array([q[0:2, 0]])
                Jn = np.concatenate((Jn0[:, 0:2], np.zeros((2, p.dim-2))), axis=1)
                dJn = np.concatenate((dJn0[:, 0:2], np.zeros((2, p.dim-2))), axis = 1)
            elif cc[i,0]>2:
                mask = np.concatenate((np.zeros((2,3)), np.ones((2, cc[i,0] -2)), np.zeros((2, p.dim - cc[i,0] -1))), axis=1)
                Jq0m = Jq0 + np.multiply(Jq0, mask)
                Jn0m = Jn0 + np.multiply(Jn0, mask)
                dJn0m = dJn0 + np.multiply(dJn0, mask)
                rn = np.transpose(np.matmul(Jq0m[:,0:2+cc[i,0]],np.ones((2+cc[i,0], 1))))
                Jn = np.concatenate((Jn0m[:, 0:2+cc[i,0]], np.zeros((2, p.n - cc[i,0]))), axis=1)
                dJn = np.concatenate((dJn0m[:, 0:2+cc[i,0]], np.zeros((2, p.n - cc[i,0]))), axis=1)
            else:
                rn = np.transpose(np.matmul(Jq0[:,0:2+cc[i,0]],np.ones((2+cc[i,0], 1))))
                Jn = np.concatenate((Jn0[:, 0:2 + cc[i, 0]], np.zeros((2, p.n - cc[i, 0]))), axis=1)
                dJn = np.concatenate((dJn0[:, 0:2 + cc[i, 0]], np.zeros((2, p.n - cc[i, 0]))), axis=1)
            rcn = rn - rc[i,:]

            # constraint Jacobian and non-Jacobian terms in constraint acceleration equation
            Jc[i,:] = np.matmul(rcn, Jn)
            xi[i,:] = np.matmul(np.matmul(np.transpose(u), np.matmul(np.transpose(Jn), Jn)), u) + np.matmul(np.matmul(rcn,dJn), u)

            # leg forces and torques

            # print("CINFO")
            # print(cInfo)
            # print(action)
            rcnn = np.sign(cInfo[i,1])*action*rcn/(np.linalg.norm(np.power(rcn,2)))
            F = F + np.transpose(Jn) @ np.array([[-rcnn[0,1]],[rcnn[0,0]]])
            F[cc[i,0]+1] = F[cc[i,0]+1] - np.sign(cInfo[i,1])*p.T

        # body bending torques
        TbA = np.concatenate(([[0]],[[0]],[[action]], [[action]], [[action]],[[0]]),axis=0)
        TbB = np.concatenate(([[0]], [[0]], [[0]], [[action]], [[action]], [[action]]), axis=0)

        if (TbA.shape == TbB.shape and TbA.shape == F.shape):
            F = F + TbA - TbB
    return [M, h, MInv, F, Jc, xi]

def impact(Jc, MInv, u):
    """compute post-impact generalized velocities"""
    up = (np.identity(len(u)) - MInv @ np.transpose(Jc) @ np.linalg.inv(Jc @ MInv @ np.transpose(Jc)) @ Jc)@u
    return up

def accelerations(q, u, F, h, MInv, Jc, xi, p):
    """compute system accelerations"""
    gamma = F + p.S@q + p.D@u -h  # external forces
    lbda = -np.linalg.inv(Jc @ MInv @ np.transpose(Jc)) @ (Jc @ MInv @ gamma + xi)  # constraint forces
    du = MInv @ (np.transpose(Jc) @ lbda + gamma)
    return du
