local PriorityQueueSimpleDecoder, Parent = 
    torch.class('nn.PriorityQueueSimpleDecoder', 'nn.Module')

function PriorityQueueSimpleDecoder:__init(inputSize, readBiasInit, 
        forgetBiasInit)
    Parent.__init(self)
    self.D = inputSize
    self.readBiasInit = readBiasInit
    self.forgetBiasInit = forgetBiasInit

    self.weight_read_in = torch.Tensor():resize(1, 1, inputSize)
    self.grad_read_in = torch.Tensor():resizeAs(self.weight_read_in)
    self.weight_read_h = torch.Tensor():resize(1, 1, inputSize)
    self.grad_read_h = torch.Tensor():resizeAs(self.weight_read_h)
    self.weight_read_b = torch.Tensor():resize(1, 1, 1)
    self.grad_read_b = torch.Tensor():resize(1, 1, 1)

    self.weight_forget_in = torch.Tensor():resize(1, 1, inputSize)
    self.grad_forget_in = torch.Tensor():resizeAs(self.weight_forget_in)
    self.weight_forget_h = torch.Tensor():resize(1, 1, inputSize)
    self.grad_forget_h = torch.Tensor():resizeAs(self.weight_forget_h)
    self.weight_forget_b = torch.Tensor():resize(1, 1, 1)
    self.grad_forget_b = torch.Tensor():resize(1, 1, 1)
    
    self.output = torch.Tensor()
    self.h_0 = torch.Tensor()
    self.R = torch.Tensor()
    self.F = torch.Tensor()

    self.rhocum = torch.Tensor()
    self.rho = torch.Tensor()
    self.phicum = torch.Tensor()
    self.phi = torch.Tensor()
    self.P = torch.Tensor()

    self.grad_Y = torch.Tensor()
    self.grad_M = torch.Tensor()


    self.grad_r = torch.Tensor()
    self.grad_f = torch.Tensor()

    self.grad_h_tp1_wrt_h_t = torch.Tensor()
    self.grad_pi_tp1 = torch.Tensor()

    self.buffer1 = torch.Tensor()
    self.buffer2 = torch.Tensor()
    self.buffer3 = torch.Tensor()
    self.buffer4 = torch.Tensor()
    self.buffer5 = torch.Tensor()

    self.maskBuffer1 = torch.ByteTensor()
    self.maskBuffer2 = torch.ByteTensor()
    self.maskBuffer3 = torch.Tensor()
    self.maskBuffer4 = torch.Tensor()
   
    self.isMaskZero = false

    self:reset()
end

function PriorityQueueSimpleDecoder:maskZero()
    self.isMaskZero = true
    return self
end

function PriorityQueueSimpleDecoder:reset()

    self.weight_read_in:uniform(-1, 1)
    self.weight_read_h:uniform(-1, 1)
    if self.readBiasInit ~= nil then
        self.weight_read_b:fill(self.readBiasInit)
    else
        self.weight_read_b:uniform(-1,1)
    end
    self.weight_forget_in:uniform(0, 1)
    self.weight_forget_h:uniform(0, 1)
    if self.forgetBiasInit ~= nil then
        self.weight_forget_b:fill(self.forgetBiasInit)
    else
        self.weight_forget_b:uniform(-1,1)
    end
    self:zeroGradParameters()

end

function PriorityQueueSimpleDecoder:zeroGradParameters()
    self.grad_read_in:zero()
    self.grad_read_h:zero()
    self.grad_read_b:zero()
    self.grad_forget_in:zero()
    self.grad_forget_h:zero()
    self.grad_forget_b:zero()
    self.grad_Y:zero()
end

function PriorityQueueSimpleDecoder:parameters()

    local params = {self.weight_read_in,
                    self.weight_read_h,
                    self.weight_read_b,
                    self.weight_forget_in,
                    self.weight_forget_h,
                    self.weight_forget_b
                   }
    local gradParams = {self.grad_read_in,
                        self.grad_read_h,
                        self.grad_read_b,
                        self.grad_forget_in,
                        self.grad_forget_h,
                        self.grad_forget_b
                       }
    return params, gradParams
end


function PriorityQueueSimpleDecoder:updateOutput(input)
    
    local isCuda = false
    if input[1]:type() == "torch.CudaTensor" then
        isCuda = true
    end

    local M, P_0, input_seq = unpack(input)
    local Tmem = M:size(1)
    local Tdec = input_seq:size(1)
    local B = input_seq:size(2)
    local D = self.D

    local R = self.R:resize(Tdec, B):zero()
    local rhocum = self.rhocum:resize(Tdec, Tmem, B):zero()
    local rho = self.rho:resize(Tdec, Tmem, B):zero()

    local F = self.F:resize(Tdec, B):zero()
    local phicum = self.phicum:resize(Tdec, Tmem, B):zero()
    local phi = self.phi:resize(Tdec, Tmem, B):zero()

    local h_0 = self.h_0:resize(B, D, 1):zero()

    local W_r_in = self.weight_read_in:expand(B, 1, D)
    local W_r_h = self.weight_read_h:expand(B, 1, D)
    local b_r = self.weight_read_b:expand(B, 1, 1)

    local W_f_in = self.weight_forget_in:expand(B, 1, D)
    local W_f_h = self.weight_forget_h:expand(B, 1, D)
    local b_f = self.weight_forget_b:expand(B, 1, 1)

    local P = self.P:resize(Tdec, Tmem, B):zero()
    P[1]:copy(P_0)

    local output = self.output:resizeAs(input_seq):zero()


    for t=1,Tdec do

        local htm1
        if t == 1 then
            htm1 = h_0
        else
            htm1 = self.output[t-1]:view(B, D, 1)
        end
        
        local yt = input_seq[t]:view(B, D, 1)
        local Rt = R[t]:view(B,1,1)
        local Ft = F[t]:view(B,1,1)
        local output_t = output[t]
        
        -- Compute remember value at step t        
        -- sigmoid(W_r_in * Y_t + W_r_h * h_tm1 + b_r)
        torch.baddbmm(Rt, b_r, W_r_in, yt)
        torch.baddbmm(Rt, Rt, W_r_h, htm1)
        torch.sigmoid(Rt, Rt)
        
        -- Compute forget value at step t        
        -- sigmoid(W_f_in * Y_t + W_f_h * h_tm1 + b_f)
        torch.baddbmm(Ft, b_f, W_f_in, yt)
        torch.baddbmm(Ft, Ft, W_f_h, htm1)
        torch.sigmoid(Ft, Ft)
        
        if self.isMaskZero then

            local mask
            if isCuda then
                mask = torch.eq(yt[{{},{1},{1}}], 0):long():nonzero()
            else
                mask = torch.eq(yt[{{},{1},{1}}], 0):nonzero()
            end
            if mask:dim() > 0 then
                for m=1,mask:size(1) do
                    Ft[mask[m][1]]:fill(0)
                    Rt[mask[m][1]]:fill(0)
                end
            end

        end

        local rhocum_t = rhocum[t]
        local rho_t = rho[t]
        local phicum_t = phicum[t]
        local phi_t = phi[t]
        
        local Pt = P[t]

        rhocum_t[{{2,Tmem},{}}]:copy(Pt[{{1,Tmem-1},{}}])
        torch.cumsum(rhocum_t, rhocum_t, 1)
        rhocum_t:csub(Rt:view(1,B):expand(Tmem,B))
        rhocum_t:mul(-1.0)
        rhocum_t:cmax(0)
        torch.cmin(rho_t, Pt, rhocum_t)

        phicum_t[{{2,Tmem},{}}]:copy(Pt[{{1,Tmem-1},{}}])
        torch.cumsum(phicum_t, phicum_t, 1)
        phicum_t:csub(Ft:view(1,B):expand(Tmem,B))
        phicum_t:mul(-1.0)
        phicum_t:cmax(0)
        torch.cmin(phi_t, Pt, phicum_t)
       
        if t+1 <= Tdec then
           torch.csub(P[t+1], Pt, phi_t) 
        end

        torch.cmul(self.buffer1, rho_t:view(Tmem,B,1):expand(Tmem,B,D), M)
        torch.sum(output_t:view(1,B,D), self.buffer1, 1)

    end
    return self.output
end

function PriorityQueueSimpleDecoder:updateGradInput(input, gradOutput)
    local M, P_0, input_seq = unpack(input)
    local Tmem = M:size(1)
    local Tdec = input_seq:size(1)
    local B = input_seq:size(2)
    local D = self.D

    local grad_Y = self.grad_Y:resizeAs(input_seq):zero()
    local grad_M = self.grad_M:resizeAs(M):zero()

    local grad_h_t_wrt_rho_t = M

    local grad_rho_wrt_r = torch.cmul(
        torch.gt(self.P, self.rhocum), 
        torch.gt(self.rhocum, 0)):typeAs(self.output)


    if self.prevMemSize ~= Tmem then
        self.lt_mask = torch.tril(torch.ByteTensor(Tmem, Tmem):fill(1), -1)
        self.lt_mask = self.lt_mask:view(1, 1, Tmem, Tmem):expand(
            Tdec, B, Tmem, Tmem)
        self.diag_mask = torch.ByteTensor(Tmem, Tmem):zero()
        for i=1,Tmem do self.diag_mask[i][i] = 1 end 
        self.diag_mask = self.diag_mask:view(1, 1, Tmem, Tmem):expand(
            Tdec, B, Tmem, Tmem)

        self.grad_wrt_pi_vals = torch.Tensor(Tmem, Tmem):typeAs(M):zero()
        self.grad_wrt_pi_vals[1][1] = 1
        for i=2,Tmem do
            self.grad_wrt_pi_vals[i][{{1,i-1}}] = -1
            self.grad_wrt_pi_vals[i][i] = 1
        end

        self.prevMemSize = Tmem
    end

    local gtmask = torch.gt(self.rhocum, 0)
    self.maskBuffer1 = self.maskBuffer1:typeAs(gtmask)
    self.maskBuffer2 = self.maskBuffer2:typeAs(gtmask)
    self.diag_mask = self.diag_mask:typeAs(gtmask)
    self.lt_mask = self.lt_mask:typeAs(gtmask)

    local rho_pi_lt_mask = torch.cmul(
        self.maskBuffer1,
        torch.gt(self.P, self.rhocum):cmul(gtmask):view(
            Tdec, Tmem, B, 1):expand(Tdec, Tmem, B, Tmem):permute(1,3,2,4),
        self.lt_mask)

    local rho_pi_d_mask = torch.cmul(
        self.maskBuffer2,
        torch.lt(self.P, self.rhocum):cmul(
            torch.gt(self.P, 0)):view(
                Tdec, Tmem, B, 1):expand(Tdec, Tmem, B, Tmem):permute(1,3,2,4),
        self.diag_mask)

    local rho_pi_mask = rho_pi_lt_mask:add(rho_pi_d_mask)
        
    local grad_rho_wrt_pi = torch.cmul(
        self.maskBuffer3,
        self.grad_wrt_pi_vals:view(1,1,Tmem, Tmem):expand(Tdec, B, Tmem, Tmem), 
        rho_pi_mask:typeAs(M))

    local phi_pi_lt_mask = torch.cmul(
        self.maskBuffer1,
        torch.gt(self.P, self.phicum):cmul(torch.gt(self.phicum, 0)):view(
            Tdec, Tmem, B, 1):expand(Tdec, Tmem, B, Tmem):permute(1,3,2,4),
        self.lt_mask)

    local phi_pi_d_mask = torch.cmul(
        self.maskBuffer2,
        torch.lt(self.P, self.phicum):cmul(
            torch.gt(self.P, 0)):view(
                Tdec, Tmem, B, 1):expand(Tdec, Tmem, B, Tmem):permute(1,3,2,4),
        self.diag_mask)

    local phi_pi_mask = phi_pi_lt_mask:add(phi_pi_d_mask)
        
    local grad_phi_wrt_pi = torch.cmul(
        self.maskBuffer4,
        self.grad_wrt_pi_vals:view(1,1,Tmem, Tmem):expand(Tdec, B, Tmem, Tmem), 
        phi_pi_mask:typeAs(M))


    local one_minus_r = self.buffer1:resizeAs(self.R):fill(1) 
    one_minus_r:csub(self.R)
    local grad_r = torch.cmul(
        self.grad_r, 
        self.R, one_minus_r):view(Tdec,1,B)
    
    local grad_phi_wrt_f = torch.cmul(
        torch.gt(self.P, self.phicum), 
        torch.gt(self.phicum, 0)):typeAs(self.output)

    local one_minus_f = self.buffer1:resizeAs(self.F):fill(1) 
    one_minus_f:csub(self.F)
    local grad_f = torch.cmul(
        self.grad_f, 
        self.F, one_minus_f):view(Tdec,1,B)
 
    local grad_f_wrt_y = self.weight_forget_in:expand(Tdec,B,D)

    local grad_r_wrt_y = self.weight_read_in:expand(Tdec,B,D)
    local grad_r_t_wrt_h_t = self.weight_read_h:view(1,D):expand(B,D)

    local grad_h_tp1_wrt_h_t = self.grad_h_tp1_wrt_h_t:resize(B,D):zero()

    local grad_pi_tp1 = self.grad_pi_tp1:resize(Tmem,B):zero()

    for t=Tdec,1,-1 do
        
        local htm1
        if t > 1 then
            htm1 = self.output[t-1]
        else
            htm1 = self.h_0:resize(B,D):zero()
        end

        local pi_t = self.P[t] -- Tmem x B
        local grad_o_t_wrt_h_t = gradOutput[t] -- path A

        -- combine path A & B
        local grad_o_t_and_h_tp1_wrt_h_t = torch.add(
            self.buffer1,
            grad_o_t_wrt_h_t, 
            grad_h_tp1_wrt_h_t)

        grad_M:add( 
            torch.cmul(
                self.buffer2,
                grad_o_t_and_h_tp1_wrt_h_t:view(1,B,D):expand(Tmem,B,D),
                self.rho[t]:view(Tmem,B,1):expand(Tmem,B,D)
            )
        )

        -- path C
        local grad_o_t_and_h_tp1_wrt_rho_t = torch.cmul(
            self.buffer2,
            grad_o_t_and_h_tp1_wrt_h_t:view(1,B,D):expand(Tmem,B,D),
            grad_h_t_wrt_rho_t)

        -- path D
        local grad_o_t_and_h_tp1_wrt_r_t = 
            torch.sum(self.buffer3, grad_o_t_and_h_tp1_wrt_rho_t, 3):cmul(
                grad_rho_wrt_r[t]:view(Tmem,B,1))
        grad_o_t_and_h_tp1_wrt_r_t = torch.cmul(
            self.buffer3,
            torch.sum(self.buffer4, grad_o_t_and_h_tp1_wrt_r_t, 1):view(B,1),
            grad_r[t]:view(B,1))

        -- path E
        local grad_o_t_and_h_tp1_wrt_y_t = torch.cmul(
            self.buffer4,
            grad_o_t_and_h_tp1_wrt_r_t:expand(B,D),
            grad_r_wrt_y[t]
        )
        grad_Y[t]:add(grad_o_t_and_h_tp1_wrt_y_t)
       
        self.grad_read_in:add(
            torch.sum(
                self.buffer1, 
                torch.cmul(
                    self.buffer4,
                    grad_o_t_and_h_tp1_wrt_r_t:expand(B,D),
                    input_seq[t]
                ), 
                1
            ):view(1,1,D)
        )

        self.grad_read_h:add(
            torch.sum(
                self.buffer1,
                torch.cmul(
                    self.buffer4,
                    grad_o_t_and_h_tp1_wrt_r_t:expand(B,D),
                    htm1
                ),
                1
            ):view(1,1,D)
        )
        
        self.grad_read_b:add(
            torch.sum(
                self.buffer1,
                grad_o_t_and_h_tp1_wrt_r_t,
                1
            ):view(1,1,1))

        -- path F
        torch.cmul(
            grad_h_tp1_wrt_h_t,
            grad_o_t_and_h_tp1_wrt_r_t:expand(B,D),
            grad_r_t_wrt_h_t)


        local phicum_t = self.phicum[t]
        local grad_o_t_and_h_tp1_wrt_pi_t = torch.bmm(
            self.buffer1:resize(B, 1, Tmem),
            grad_o_t_and_h_tp1_wrt_rho_t:sum(3):permute(2,3,1),
            grad_rho_wrt_pi[t]):squeeze(2):t()

        -- path H
        local grad_pi_tp1_wrt_phi_t = 
            torch.mul(
                self.buffer2,
                grad_pi_tp1, -1.0)
        
        local grad_pi_tp1_wrt_pi_t = 
            torch.bmm( 
                self.buffer3:resize(B, 1, Tmem),
                grad_pi_tp1_wrt_phi_t:view(Tmem, 1, B):permute(3,2,1),
                grad_phi_wrt_pi[t])
        grad_pi_tp1_wrt_pi_t = grad_pi_tp1_wrt_pi_t:squeeze(2):t()

        grad_pi_tp1:add(grad_o_t_and_h_tp1_wrt_pi_t)

        --path I
        local grad_pi_tp1_wrt_ft = torch.cmul(
            self.buffer1,
            grad_pi_tp1_wrt_phi_t,
            grad_phi_wrt_f[t])
        grad_pi_tp1_wrt_ft = torch.cmul(
            self.buffer1,
            torch.sum(self.buffer2, grad_pi_tp1_wrt_ft, 1),
            grad_f[t])
        
        -- path J
        grad_Y[t]:add( 
            torch.cmul(
                self.buffer2,
                grad_pi_tp1_wrt_ft:view(B,1):expand(B,D),
                grad_f_wrt_y[t]))
        
        self.grad_forget_in:add(
            torch.sum(
                self.buffer4,
                torch.cmul(
                    self.buffer2,
                    grad_pi_tp1_wrt_ft:view(B,1):expand(B,D),
                    input_seq[t]),
                1):view(1,1,D))

        self.grad_forget_h:add(
            torch.sum(
                self.buffer4,
                torch.cmul(
                    self.buffer2,
                    grad_pi_tp1_wrt_ft:view(B,1):expand(B,D),
                    htm1),
                1):view(1,1, D))

        self.grad_forget_b:add(
            torch.sum(
                self.buffer4,
                grad_pi_tp1_wrt_ft:view(B,1),
                1):view(1,1,1))

        local grad_pi_tp1_wrt_ht = torch.cmul(
            self.buffer2,
            grad_pi_tp1_wrt_ft:view(B,1):expand(B,D),
            self.weight_forget_h:view(1,D):expand(B,D))
        
        grad_h_tp1_wrt_h_t:add(grad_pi_tp1_wrt_ht)

        -- path K
        grad_pi_tp1:add(grad_pi_tp1_wrt_pi_t)


    end

    if self.isMaskZero then
        grad_pi_tp1[torch.eq(M:select(3,1), 0)] = 0
    end

    self.gradInput = {grad_M, grad_pi_tp1, grad_Y}

    return self.gradInput
end

function PriorityQueueSimpleDecoder:accGradParameters(input, gradOutput)
   

end


