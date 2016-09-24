local PriorityQueueSimpleDecoder, Parent = 
    torch.class('nn.PriorityQueueSimpleDecoder', 'nn.Module')

function PriorityQueueSimpleDecoder:__init(inputSize)
    Parent.__init(self)
    self.D = inputSize

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

    self.buffer1 = torch.Tensor()
    self.buffer2 = torch.Tensor()
    self.buffer3 = torch.Tensor()
    self.buffer4 = torch.Tensor()
    self.buffer5 = torch.Tensor()
    self.buffer6 = torch.Tensor()
    self.buffer7 = torch.Tensor()
    self.buffer8 = torch.Tensor()
    self.buffer9 = torch.Tensor()
    self.buffer10 = torch.Tensor()
    self.buffer11 = torch.Tensor()
    self.buffer12 = torch.Tensor()
    self.buffer13 = torch.Tensor()
    self.buffer14 = torch.Tensor()
    self.buffer15 = torch.Tensor()
    self.buffer16 = torch.Tensor()
    self.buffer17 = torch.Tensor()

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
    self.weight_read_b:fill(1) -- initial bias is to read
    self.weight_forget_in:uniform(-1, 1)
    self.weight_forget_h:uniform(-1, 1)
    self.weight_forget_b:fill(-3) -- initial bias is to remember

    self:zeroGradParameters()

end


function PriorityQueueSimpleDecoder:updateOutput(input)
    
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
        torch.bmm(Rt, W_r_in, yt)
        torch.add(Rt, Rt, torch.bmm(W_r_h, htm1))
        --torch.baddbmm(r[t], r[t], W_r_h, htm1)
        Rt:add(b_r)
        torch.sigmoid(Rt, Rt)
        
        -- Compute forget value at step t        
        -- sigmoid(W_f_in * Y_t + W_f_h * h_tm1 + b_f)
        torch.bmm(Ft, W_f_in, yt)
        torch.add(Ft, Ft, torch.bmm(W_f_h, htm1))
        Ft:add(b_f)
        torch.sigmoid(Ft, Ft)
        
        if self.isMaskZero then
            local mask = torch.eq(yt[{{},{1},{1}}], 0):nonzero()
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
           P[t+1]:copy(Pt - phi_t) 
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

    local grad_W_ry = 
        torch.Tensor():resizeAs(input_seq):typeAs(input_seq):zero()
    local grad_W_rh =
        torch.Tensor():resizeAs(input_seq):typeAs(input_seq):zero()
    local grad_b_r =
        torch.Tensor():resize(Tdec,B,1):typeAs(input_seq):zero()

    local grad_W_fy = 
        torch.Tensor():resizeAs(input_seq):typeAs(input_seq):zero()
    local grad_W_fh = 
        torch.Tensor():resizeAs(input_seq):typeAs(input_seq):zero()
    local grad_b_f =
        torch.Tensor():resize(Tdec,B,1):typeAs(input_seq):zero()

    local grad_Y = self.grad_Y:resizeAs(input_seq):zero()
    local grad_M = self.grad_M:resizeAs(M):zero()

    local grad_h_t_wrt_rho_t = M

    local grad_rho_wrt_r = torch.cmul(
        torch.gt(self.P, self.rhocum), 
        torch.gt(self.rhocum, 0)):typeAs(self.output)

    local grad_r = torch.cmul(self.R, 1.0 - self.R):view(Tdec,1,B)
    
    local grad_phi_wrt_f = torch.cmul(
        torch.gt(self.P, self.phicum), 
        torch.gt(self.phicum, 0)):typeAs(self.output)
    local grad_f = torch.cmul(self.F, 1.0 - self.F):view(Tdec,1,B)
    local grad_f_wrt_y = self.weight_forget_in:expand(Tdec,B,D)

    local grad_r_wrt_y = self.weight_read_in:expand(Tdec,B,D)
    local grad_r_t_wrt_h_t = self.weight_read_h:view(1,D):expand(B,D)

    local grad_h_tp1_wrt_h_t = 
        torch.Tensor():resize(B,D):typeAs(self.h_0):zero()


    local grad_pi_tp1 = 
        torch.Tensor():resize(Tmem,B):typeAs(self.output):zero()

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
            grad_o_t_wrt_h_t, grad_h_tp1_wrt_h_t)

        grad_M = grad_M + 
            torch.cmul(
                grad_o_t_and_h_tp1_wrt_h_t:view(1,B,D):expand(Tmem,B,D),
                self.rho[t]:view(Tmem,B,1):expand(Tmem,B,D)
            )

        -- path C
        local grad_o_t_and_h_tp1_wrt_rho_t = torch.cmul(
            grad_o_t_and_h_tp1_wrt_h_t:view(1,B,D):expand(Tmem,B,D),
            grad_h_t_wrt_rho_t)

        -- path D
        local grad_o_t_and_h_tp1_wrt_r_t = torch.cmul(
            grad_o_t_and_h_tp1_wrt_rho_t:sum(3),
            grad_rho_wrt_r[t]:view(Tmem,B,1))
        grad_o_t_and_h_tp1_wrt_r_t = torch.cmul(
            grad_o_t_and_h_tp1_wrt_r_t:sum(1):view(B,1),
            grad_r[t]:view(B,1))

        -- path E
        local grad_o_t_and_h_tp1_wrt_y_t = torch.cmul(
            grad_o_t_and_h_tp1_wrt_r_t:expand(B,D),
            grad_r_wrt_y[t]
        )
        grad_Y[t]:copy(grad_o_t_and_h_tp1_wrt_y_t)
       

        grad_W_ry[t] = 
            torch.cmul(
                grad_o_t_and_h_tp1_wrt_r_t:expand(B,D),
                input_seq[t]
            )

        grad_W_rh[t] = 
            torch.cmul(
                grad_o_t_and_h_tp1_wrt_r_t:expand(B,D),
                htm1
            )
        
        grad_b_r[t]:copy(grad_o_t_and_h_tp1_wrt_r_t)

        -- path F
        torch.cmul(
            grad_h_tp1_wrt_h_t,
            grad_o_t_and_h_tp1_wrt_r_t:expand(B,D),
            grad_r_t_wrt_h_t)


        local rhocum_t = self.rhocum[t]
        local phicum_t = self.phicum[t]

        -- path G
        local grad_o_t_and_h_tp1_wrt_pi_t = 
            torch.Tensor():resize(Tmem,B):typeAs(self.output):zero()
        for j=1,Tmem do
            local grad_rho_j_wrt_pi_j = torch.cmul(
                torch.le(pi_t[j], rhocum_t[j]),
                torch.gt(rhocum_t[j], 0)):typeAs(self.output)
            grad_o_t_and_h_tp1_wrt_pi_t[j]:add(
                torch.cmul(
                    grad_rho_j_wrt_pi_j,
                    grad_o_t_and_h_tp1_wrt_rho_t[j]:sum(2):view(B)))
            for i=j+1,Tmem do
                local grad_rho_i_wrt_pi_j = torch.cmul(
                    torch.gt(pi_t[i], rhocum_t[i]),
                    torch.gt(rhocum_t[i], 0)):typeAs(self.output):mul(-1)
                    
                grad_o_t_and_h_tp1_wrt_pi_t[j]:add(
                    torch.cmul(
                        grad_rho_i_wrt_pi_j,
                        grad_o_t_and_h_tp1_wrt_rho_t[i]:sum(2):view(B)))
                    
            end
        end
      

        -- path H
        local grad_pi_tp1_wrt_phi_t = 
            grad_pi_tp1 * -1.0
        local grad_pi_tp1_wrt_pi_t = 
            torch.Tensor():resize(Tmem,B):typeAs(self.output):zero()
        for j=1,Tmem do
            local grad_phi_j_wrt_pi_j = torch.cmul(
                torch.le(pi_t[j], phicum_t[j]),
                torch.gt(phicum_t[j], 0)):typeAs(self.output)
            grad_pi_tp1_wrt_pi_t[j]:add(
                torch.cmul(
                    grad_pi_tp1_wrt_phi_t[j],
                    grad_phi_j_wrt_pi_j))
            for i=j+1,Tmem do
                local grad_phi_i_wrt_pi_j = torch.cmul(
                    torch.gt(pi_t[i], phicum_t[i]),
                    torch.gt(phicum_t[i], 0)):typeAs(self.output):mul(-1)
                    
                grad_pi_tp1_wrt_pi_t[j]:add(
                    torch.cmul(
                        grad_phi_i_wrt_pi_j,
                        grad_pi_tp1_wrt_phi_t[i]))
                    
            end
        end
 


        --path I
        local grad_pi_tp1_wrt_ft = torch.cmul(
            grad_pi_tp1_wrt_phi_t,
            grad_phi_wrt_f[t])
        grad_pi_tp1_wrt_ft = torch.cmul(
            grad_pi_tp1_wrt_ft:sum(1),
            grad_f[t])
        
        -- path J
        grad_Y[t] = grad_Y[t] + 
            torch.cmul(
                grad_pi_tp1_wrt_ft:view(B,1):expand(B,D),
                grad_f_wrt_y[t])
        
        grad_W_fy[t] = torch.cmul(
                grad_pi_tp1_wrt_ft:view(B,1):expand(B,D),
                input_seq[t])

        grad_W_fh[t] = torch.cmul(
            grad_pi_tp1_wrt_ft:view(B,1):expand(B,D),
            htm1)

        grad_b_f[t]:copy(grad_pi_tp1_wrt_ft:view(B,1))

        local grad_pi_tp1_wrt_ht = torch.cmul(
            grad_pi_tp1_wrt_ft:view(B,1):expand(B,D),
            self.weight_forget_h:view(1,D):expand(B,D))
        
        grad_h_tp1_wrt_h_t = grad_h_tp1_wrt_h_t + grad_pi_tp1_wrt_ht

        -- path K
        grad_pi_tp1 = grad_pi_tp1 + grad_o_t_and_h_tp1_wrt_pi_t + grad_pi_tp1_wrt_pi_t


    end


    self.grad_read_in:copy(grad_W_ry:sum(2):sum(1))
    self.grad_read_h:copy(grad_W_rh:sum(2):sum(1))
    self.grad_read_b:copy(grad_b_r:sum(2):sum(1))
    self.grad_forget_in:copy(grad_W_fy:sum(2):sum(1))
    self.grad_forget_h:copy(grad_W_fh:sum(2):sum(1))
    self.grad_forget_b:copy(grad_b_f:sum(2):sum(1))

    self.gradInput = {grad_M, grad_pi_tp1, grad_Y}

    return self.gradInput
end

function PriorityQueueSimpleDecoder:accGradParameters(input, gradOutput)
   

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

